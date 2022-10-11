import numpy as np
import torch
from pytorch_lightning import LightningModule
from .model_utils import make_mlp
from .graph_network import *
from torch_geometric.data import Batch, DataLoader
import torch.nn.functional as F

class EPDLinearObsModel(LightningModule):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        super().__init__()
        self.save_hyperparameters(cfg_dict)

        self.lr = cfg_dict['lr']
        self.batch_size = cfg_dict['batch_size']

        self.m = cfg_dict['D_U']
        self.n = cfg_dict["D_S"]
        self.T = cfg_dict['T']
        self._N_O = cfg_dict['N_O']
        self._N_R = cfg_dict['N_R']
        self._D_S = cfg_dict["D_S"]
        self._D_R = cfg_dict["D_R"]

        self._train_ds_gen = train_ds_gen
        self._val_ds_gen = val_ds_gen

        self._cfg_dict_PN = {
            'D_S_in': cfg_dict['D_S'],
            'D_R_in': cfg_dict['D_R'],
            'D_G_in': cfg_dict['D_G'],
            'D_S_out': cfg_dict['D_S'],
            'D_R_out': cfg_dict['D_R'],
            'D_G_out': cfg_dict['D_G']
        }

        gpu_load = cfg_dict.get('gpu', 0)
        self._device = f'cuda:{gpu_load}'

        self._cfg_dict = cfg_dict
        self._make_model()

    @property
    def ds(self):
        return self._train_ds_gen()
    
    @property
    def val_ds(self):
        return self._val_ds_gen()

    def _make_model(self):
        self._processor_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                for i in range(self._cfg_dict['num_message_passing_steps'])])

        self._linear_model_A_network = make_mlp(
            self._cfg_dict['D_S'], self._cfg_dict['hidden_layers_A'] + [self.n*3], 
            prefix='linear_model_A_network', last_act='Sigmoid'
            ).to(self._device)

        self._linear_model_B_network = make_mlp(
            self._cfg_dict['D_S'], self._cfg_dict['hidden_layers_B'] + [self.n*self.m], 
            prefix='linear_model_B_network', last_act='Sigmoid'
            ).to(self._device)

    def _process(self, processor_networks, latent_graphs_0, h0=None, T=None):
        """Processes the latent graph with several steps of message passing."""
        latent_graphs_prev_k = latent_graphs_0.clone()
        for processor_network_k in processor_networks:
            latent_graphs_k = self._process_step(
                processor_network_k, latent_graphs_prev_k)
            latent_graphs_prev_k = latent_graphs_k.clone()
        return latent_graphs_k

    def _process_step(self, processor_network_k, latent_graphs_prev_k):
        """Single step of message passing with node/edge residual connections."""

        # One step of message passing.
        latent_graphs_k = processor_network_k(latent_graphs_prev_k)
        
        # Add residuals.
        latent_graphs_k.pos = latent_graphs_k.pos + latent_graphs_prev_k.pos 
        latent_graphs_k.edge_attr = latent_graphs_k.edge_attr + latent_graphs_prev_k.edge_attr 

        return latent_graphs_k
    
    def _predict_next_state_via_linear_model(self, input_graphs):
        processed_graphs = self._process(self._processor_networks, input_graphs)
        r = self._linear_model_A_network(processed_graphs.pos)
        P = torch.eye(self.n).to(self._device)[None,:,:] + torch.matmul(r[:,:self.n,None], r[:,None,self.n:2*self.n])
        D = torch.diag_embed(r[:,2*self.n:3*self.n])
        A = torch.linalg.solve(P, D)@P

        B = self._cfg_dict["scale_B"]*self._linear_model_B_network(processed_graphs.pos).view(-1,self.n,self.m)

        control = processed_graphs.control[:,:,None]

        xtp1 = (torch.matmul(A, input_graphs.pos[:,:self.n,None]) + torch.matmul(B, control)).view(-1,self.n)
        
        return xtp1, A, B, None, None

    def forward(self, input_graphs):
        xtp1, A, B, offset, params_out = self._predict_next_state_via_linear_model(input_graphs)
        if torch.any(torch.isnan(xtp1)):
            import ipdb; ipdb.set_trace()
        return xtp1, params_out

    def training_step(self, batch, batch_idx):
        xtp1, _ = self(batch)

        loss = 1e+4*F.mse_loss(xtp1, batch.y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx): 
        xtp1, _ = self(batch)

        loss = 1e+4*F.mse_loss(xtp1, batch.y)

        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
    def process_input_and_final_state(self, x0, xf):
        dataz0 = Batch.from_data_list([self.ds.data_from_input(x0, np.zeros(self.m)).to(self._device)])
        datazf = Batch.from_data_list([self.ds.data_from_input(xf, np.zeros(self.m)).to(self._device)])
        z0 = dataz0.pos[:,:self.n].flatten()[:,None].cpu().numpy()
        zf = datazf.pos[:,:self.n].flatten()[:,None].cpu().numpy()
        return dataz0, datazf, z0, zf
    
    def _post_process_tau_batch_to_np(self, tau_batch, n):
        tau = np.zeros((tau_batch.num_graphs, n*self._N_O+self.m, 1))
        for i in range(tau_batch.num_graphs):
            tau[i, :n*self._N_O, 0] = tau_batch.pos[i*self._N_O:(i+1)*self._N_O, :n].flatten().cpu().numpy()
            tau[i, n*self._N_O:, 0] = tau_batch.control[i*self._N_O].cpu().numpy()
        return tau

    def _post_process_tau_data_list_to_np(self, data_list, n):
        n_graphs = len(data_list)
        tau = np.zeros((n_graphs, n*self._N_O+self.m, 1))
        for i in range(n_graphs):
            tau[i, :n*self._N_O, 0] = data_list[i].pos[:, :n].flatten().cpu().numpy()
            tau[i, n*self._N_O:, 0] = data_list[i].control[0].cpu().numpy()
        return tau
    
    def post_process_tau_latent_to_obs(self, tau, n, is_numpy=True):
        if is_numpy:
            return tau
        else:
            return self._post_process_tau_batch_to_np(tau, n)
    
    def linearize_dynamics_obs_space(self, tau):
        tau_graphs = self.post_process_tau_np_to_batch(tau, is_latent=False)
        _, A, B, offset, _ = self._predict_next_state_via_linear_model(tau_graphs)
        F, f = self.post_process_ABC(A, B, offset)
        return F, f
    
    def post_process_ABC(self, A, B, offset):
        # Converts A,B,C from model output format to ltv_lqr() input format
        len_A= A.shape[0]

        A = [A[torch.arange(i, len_A, self._N_O),:] for i in range(self._N_O)]
        B = [B[torch.arange(i, len_A, self._N_O),:] for i in range(self._N_O)]
        offset = [offset[i::self._N_O,:] for i in range(self._N_O)]
        
        AC = [A[i].view(-1, self.n, self.n) for i in range(self._N_O)]
        batch_size = AC[0].shape[0]
        AC = [torch.block_diag(*[AC[node][i] for node in range(self._N_O)]) for i in range(batch_size)]
        AC = torch.stack(AC, axis=0)
        B = torch.cat(B, axis=1)
        offset = torch.cat(offset, axis=1)
        
        F = torch.cat([AC, B.view(-1, self.n*self._N_O, self.m)], axis=2).cpu().numpy()
        # f = np.zeros((F.shape[0], self.n*self._N_O, 1))
        f = offset[:,:,None].cpu().numpy()
        return F, f

    def post_process_tau_np_to_batch(self, tau, is_latent=True):
        # Converts tau from ltv_lqr() output format to model input format, no variance
        # tau.shape = (batch_size, n*n_nodes + m, 1)
        batch_size = tau.shape[0]
        n = (tau.shape[1] - self.m) // self._N_O

        if is_latent:
            dataz = self.convert_to_latent_state().to_data_list()[0] # TODO: latent state should have info of edge and global attributes

        data = []
        for i in range(batch_size):
            if not is_latent:
                datat = self.ds.data_from_input(tau[i, :n*self._N_O, 0], tau[i, n*self._N_O:, 0])
            else:
                import ipdb; ipdb.set_trace()
                datat = dataz.clone()
                datat.pos = torch.tensor(tau[i, :n*self._N_O, :].reshape(self._N_O, n), dtype=torch.float32)
                datat.control = torch.tensor(np.repeat(tau[i, n*self._N_O:, :], self._N_O, axis=1).T, dtype=torch.float32)
            data.append(datat)
            datat = datat.clone()
        data = Batch.from_data_list(data).to(self._device)
        return data
    
    def forward_propagate_control_lqr(self, z0, control, return_numpy_dyn=True, ht=None, return_mode=False):
        '''
        Inputs:
            control: Dict('K', 'k'). 'K': numpy array of size=(T, m, n), 'k': numpy array of size=(T, m, 1)
            z0: latent state in batch format
        '''

        T = control['K'].shape[0]
        zt = z0.clone()
        K = torch.tensor(control['K'], dtype=torch.float32).to(self._device)
        k = torch.tensor(control['k'], dtype=torch.float32).to(self._device)
        data, A_full, B_full, offset_full = [], [], [], []
        mode = []
        for t in range(T):
            action = K[t]@zt.pos.flatten().view(self.n*self._N_O,1) + k[t]
            zt.control = action.repeat(1, self._N_O).T
            if self._cfg_dict["D_S"] == 4: # TODO: remove hardcoded stuff
                zt.control[1::self._N_O, :] = 0.0
            data.extend(zt.to_data_list())
            zt_pos, A, B, offset, _ = self._predict_next_state_via_linear_model(zt, ht=ht, T=1)

            if return_mode:
                mode.append(self._edge_act_onehot)
            if torch.any(torch.isnan(zt.pos)):
                import ipdb; ipdb.set_trace()
            A_full.append(A)
            B_full.append(B)
            offset_full.append(offset)
            zt.pos = zt_pos.clone()
            _z = self.ds.data_from_input(zt_pos.flatten().cpu().numpy(), np.zeros(self.m), self.n).to(self._device)
            zt.edge_attr = _z.edge_attr.clone()
            if ht is not None:
                ht = self.update_ht_fwd_prop(ht, zt)
        tau_batch = Batch.from_data_list(data)
        A_full = torch.cat(A_full, axis=0)
        B_full = torch.cat(B_full, axis=0)
        offset_full = torch.cat(offset_full, axis=0)
        if return_mode:
            mode = torch.stack(mode,axis=0)
        if return_numpy_dyn:
            F, f = self.post_process_ABC(A_full, B_full, offset_full)
            return tau_batch, F, f, mode
        else:
            return tau_batch, A_full, B_full, offset_full
            
    
    def _progress_fn(self, input, batch_size, n):
        input = input.view(batch_size, n*self.T, -1)
        input = torch.cat([input[:, n:, :], 0 * input[:, :n, :]], axis=1)
        return input.view(batch_size*n*self.T, -1)

    def forward_propagate_control(self, z0, control, ht=None, return_mode=False):
        '''
        Inputs:
            control: shape=(T,m)
            z0: state in batch format
        '''
        T = control.shape[0]
        zt = z0.clone()
        control = torch.tensor(control, dtype=torch.float32).view(T,-1,1).to(self._device)
        data = []
        mode = []
        for t in range(T):
            zt.control = control[t].repeat(1,self._N_O).T
            data.extend(zt.to_data_list())
            
            if t>0 and ht is not None:
                ht = htp1.clone()
            zt_pos, *_ = self._predict_next_state_via_linear_model(zt, ht=ht, T=1)

            if return_mode:
                mode.append(self._edge_act_onehot)
            if torch.any(torch.isnan(zt.pos)):
                import ipdb; ipdb.set_trace()

            zt.pos = zt_pos.clone()
            _z = self.ds.data_from_input(zt_pos.flatten().cpu().numpy(), np.zeros(self.m), self.n).to(self._device)
            zt.edge_attr = _z.edge_attr.clone()
            if ht is not None:
                htp1 = self.update_ht_fwd_prop(ht, zt)
        
        tau_batch = Batch.from_data_list(data)
        if return_mode:
            mode = torch.stack(mode,axis=0)
        
        return tau_batch, ht

class EPDLinearObsSpringMassDamperModel(EPDLinearObsModel):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        super(EPDLinearObsSpringMassDamperModel, self).__init__(train_ds_gen=train_ds_gen, val_ds_gen=val_ds_gen, cfg_dict=cfg_dict)

    def _make_model(self):
        n_dyn_params_node = self._cfg_dict['dof']*5 if self._cfg_dict["learn_impact_params"] else self._cfg_dict['dof']*4

        if self._cfg_dict["heterogeneous_nodes_pn"]:
            self._processor_networks = torch.nn.ModuleList([GraphNetworkHeteroNodes({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])
        else:
            self._processor_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])

        self._linear_model_parameters = make_mlp(
            self._cfg_dict_PN['D_S_out'], self._cfg_dict['hidden_layers_mkc'] + [n_dyn_params_node], 
            prefix='linear_model_parameters_network', act='relu', last_act='sigmoid'
            ).to(self._device)

    def _predict_next_state_via_linear_model(self, input_graphs, ht=None, T=None):
        dt = self._cfg_dict["dt"]
        dof = self._cfg_dict['dof']

        processed_graphs = self._process(self._processor_networks, input_graphs, h0=ht, T=T)

        params = self._linear_model_parameters(processed_graphs.pos) # [1/mx, 1/my, kx, ky, cx, cy, x0, y0, ix, iy, ixo, iyo]
        # mkc = self._cfg_dict['scale_params']*torch.sigmoid(params[:,:6])
        # xy_0 = self._cfg_dict['scale_params']*torch.tanh(params[:,6:8])

        mkc = self._cfg_dict['scale_params']*params[:,:dof*3]

        if self._cfg_dict['dataset_name'] == 'EPDDoorOpeningBox2D1DoorDataset':
            xy_0 = input_graphs.hinge_loc.clone()
        else:
            xy_0 = self._cfg_dict['scale_params']*params[:,dof*3:dof*4]

        if self._cfg_dict["heterogeneous_nodes_pn"]:
            B_part1 = torch.cat((
                            torch.zeros((dof, dof)),
                            dt*torch.eye(dof)
                            ), axis=0).to(self._device)
            
            A_part1 = torch.cat((
                        torch.cat((torch.eye(dof), dt*torch.eye(dof)), axis=1),
                        torch.zeros((dof, 2*dof))
                        ), axis=0).to(self._device)
        else:
            B_part1 = torch.tensor([[0., 0.],
                                    [0., 0.],
                                    [dt, 0.],
                                    [0., dt],
                                    [0., 0.]]).to(self._device)
            
            A_part1 = torch.tensor([[1., 0., dt, 0., 0.],
                                    [0., 1., 0., dt, 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]]).to(self._device)

        A = torch.stack([A_part1]*params.shape[0], 0)

        B_part2 = torch.diag_embed(mkc[:,:dof]) # 1/mx, 1/my
        B = B_part1@B_part2

        offset = torch.zeros(input_graphs.pos.shape).to(self._device)
        offset[:, dof:2*dof] = mkc[:,:dof]*mkc[:,dof:2*dof]*xy_0*dt # k*x0*dt/m

        k_by_m = torch.diag_embed(-1.*mkc[:,:dof]*mkc[:,dof:2*dof]*dt)
        if self._cfg_dict["learn_impact_params"]:
            ixy = self._cfg_dict['scale_params']*params[:,dof*4:dof*5]
            # ixy = torch.sigmoid(params[:,8:10])
            # ixy_off = self._cfg_dict['scale_params']*torch.tanh(params[:,10:12])
            i_c_by_m = torch.diag_embed( ixy*(1. - (mkc[:,:dof]*mkc[:,dof*2:dof*3]*dt)) )
            i_dt = torch.diag_embed(ixy*dt)
            A[:, :dof, dof:2*dof] = i_dt.clone()
            
            # offset[:, :2] = ixy_off*dt
            # offset[:, 2:4] = mkc[:,:2]*mkc[:,2:4]*xy_0*dt + (1. - (mkc[:,:2]*mkc[:,4:6]*dt))*ixy_off # k*x0*dt/m + (1-c*dt/m)*io
            # params_out = torch.cat((mkc, xy_0, ixy, ixy_off), 1)
            params_out = torch.cat((mkc, xy_0, ixy), 1)
        else:
            i_c_by_m = torch.diag_embed( 1. - (mkc[:,:dof]*mkc[:,dof*2:dof*3]*dt) )
            params_out = torch.cat((mkc, xy_0), 1)

        A_part2 = torch.cat((k_by_m, i_c_by_m), 2)
        A[:, dof:2*dof, :2*dof] = A_part2.clone()
        
        control = processed_graphs.control[:,:,None]
        xtp1 = (torch.matmul(A, input_graphs.pos[:,:self.n,None]) + torch.matmul(B, control)).view(-1,self.n) + offset

        return xtp1, A, B, offset, params_out

class FrankaEPDLinearObsSpringMassDamperModel(EPDLinearObsSpringMassDamperModel):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        self._dof = cfg_dict['dof']
        self._dt = cfg_dict["dt"]
        super(FrankaEPDLinearObsSpringMassDamperModel, self).__init__(train_ds_gen=train_ds_gen, val_ds_gen=val_ds_gen, cfg_dict=cfg_dict)
    
    def _make_model(self):
        self._n_dyn_params_node = self._dof*5 if self._cfg_dict["learn_impact_params"] else self._dof*4
        
        if self._cfg_dict["heterogeneous_nodes_pn"]:
            self._processor_networks = torch.nn.ModuleList([GraphNetworkHeteroNodes({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])
        else:
            self._processor_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])

        self._linear_model_parameters = make_mlp(
            self._cfg_dict_PN['D_S_out'], self._cfg_dict['hidden_layers_mkc'] + [self._n_dyn_params_node], 
            prefix='linear_model_parameters_network', act='relu', last_act='sigmoid'
            ).to(self._device)

    def _predict_next_state_via_linear_model(self, input_graphs):
        processed_graphs = self._process(self._processor_networks, input_graphs)

        params = self._linear_model_parameters(processed_graphs.pos) # [1/m, k, c, x0, i, io]*7
        # mkc = self._cfg_dict['scale_params']*torch.sigmoid(params[:,:6])
        # xy_0 = self._cfg_dict['scale_params']*torch.tanh(params[:,6:8])

        mkc = self._cfg_dict['scale_params']*params[:,:self._dof*3]
        xy_0 = self._cfg_dict['scale_params']*params[:,self._dof*3:self._dof*4]

        B_part1 = torch.cat((
            torch.zeros((self._dof, self._dof)),
            self._dt*torch.eye(self._dof)
        ), axis=0).to(self._device)
        
        A_part1 = torch.cat((
            torch.cat((torch.eye(self._dof), self._dt*torch.eye(self._dof)), axis=1),
            torch.zeros((self._dof, 2*self._dof))
            ), axis=0).to(self._device)
        
        A = torch.stack([A_part1]*params.shape[0], 0)

        B_part2 = torch.diag_embed(mkc[:,:self._dof]) # [1/m]*7
        B = B_part1@B_part2

        offset = torch.zeros(processed_graphs.pos.shape).to(self._device)
        offset[:, self._dof:2*self._dof] = mkc[:,:self._dof]*mkc[:,self._dof:2*self._dof]*xy_0*self._dt # k*x0*dt/m

        k_by_m = torch.diag_embed(-1.*mkc[:,:self._dof]*mkc[:,self._dof:2*self._dof]*self._dt)
        if self._cfg_dict["learn_impact_params"]:
            ixy = self._cfg_dict['scale_params']*params[:,self._dof*4:self._dof*5]
            # ixy = torch.sigmoid(params[:,self._dof*4:self._dof*5])
            # ixy_off = self._cfg_dict['scale_params']*torch.tanh(params[:,self._dof*5:self._dof*6])
            i_c_by_m = torch.diag_embed( ixy*(1. - (mkc[:,:self._dof]*mkc[:,self._dof*2:self._dof*3]*self._dt)) )
            i_dt = torch.diag_embed(ixy*self._dt)
            A[:, :self._dof, self._dof:2*self._dof] = i_dt.clone()
            
            # offset[:, :2] = ixy_off*self._dt
            # offset[:, 2:4] = mkc[:,:2]*mkc[:,2:4]*xy_0*self._dt + (1. - (mkc[:,:2]*mkc[:,4:6]*self._dt))*ixy_off # k*x0*dt/m + (1-c*dt/m)*io
            # params_out = torch.cat((mkc, xy_0, ixy, ixy_off), 1)
            params_out = torch.cat((mkc, xy_0, ixy), 1)
        else:
            i_c_by_m = torch.diag_embed( 1. - (mkc[:,:self._dof]*mkc[:,self._dof*2:self._dof*3]*self._dt) )
            params_out = torch.cat((mkc, xy_0), 1)

        A_part2 = torch.cat((k_by_m, i_c_by_m), 2)
        A[:, self._dof:2*self._dof, :2*self._dof] = A_part2.clone()
        
        control = processed_graphs.control[:,:,None]

        xtp1 = (torch.matmul(A, input_graphs.pos[:,:self.n,None]) + torch.matmul(B, control)).view(-1,self.n) + offset

        return xtp1, A, B, offset, params_out


class EPDLinearObsSpringMassDamperModelwithModes(EPDLinearObsSpringMassDamperModel):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        super(EPDLinearObsSpringMassDamperModelwithModes, self).__init__(train_ds_gen=train_ds_gen, val_ds_gen=val_ds_gen, cfg_dict=cfg_dict)
        assert self._cfg_dict["num_edge_types"] == len(self._cfg_dict["prior"])
        self.gumbel_tau = self._cfg_dict["gumbel_temp"]

    def _make_model(self):
        n_dyn_params_node = self._cfg_dict['dof']*5 if self._cfg_dict["learn_impact_params"] else self._cfg_dict['dof']*4
        # Message passing for forward dynamics
        if self._cfg_dict["heterogeneous_nodes_pn"]:
            self._processor_networks = torch.nn.ModuleList([GraphNetworkHeteroNodes({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])
        else:
            self._processor_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])
        
        # Message passing before edge prediction
        self._inference_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                for i in range(self._cfg_dict['num_message_passing_steps_graphInf'])])

        # MLP for edges - edge activation update
        self._fc_phi_R_act = make_mlp(
                            self._cfg_dict_PN["D_R_out"], self._cfg_dict["hidden_layers_phi_R"] + [self._cfg_dict["num_edge_types"]],
                            prefix='fc_phi_R_act', last_act='relu'
                        ).to(self._device)

        self._linear_model_parameters = make_mlp(
            self._cfg_dict_PN['D_S_out'], self._cfg_dict['hidden_layers_mkc'] + [n_dyn_params_node], 
            prefix='linear_model_parameters_network', act='relu', last_act='sigmoid'
            ).to(self._device)

    def _process(self, processor_networks, latent_graphs_0, h0=None, T=None):
        """Processes the latent graph with several steps of message passing."""

        latent_graphs_prev_k = latent_graphs_0.clone()
        for inference_network_k in self._inference_networks:
            latent_graphs_k = self._process_step(
                inference_network_k, latent_graphs_prev_k)
            latent_graphs_prev_k = latent_graphs_k.clone()
        
        self._edge_act_logits = self._fc_phi_R_act(latent_graphs_prev_k.edge_attr)

        if self._cfg_dict['sampling'] == 'argmax':
            self._edge_act = torch.softmax(self._edge_act_logits, dim=1)
            
            indx = self.edge_act1.max(dim=1)[1].view(-1,1)
            edge_act_hard = torch.zeros_like(self._edge_act)
            edge_act_hard.scatter_(dim=1, index=indx, src=torch.ones_like(self._edge_act))
            self._edge_act_onehot = (edge_act_hard - self._edge_act).detach() + self._edge_act

        if self._cfg_dict['sampling'] == 'gumbel':
            self._edge_act_onehot = F.gumbel_softmax(self._edge_act_logits, tau=1, hard=True)

        latent_graphs_prev_k = latent_graphs_0.clone()
        latent_graphs_prev_k.edge_act = self._edge_act_onehot.clone()
        for processor_network_k in processor_networks:
            latent_graphs_k = self._process_step(
                processor_network_k, latent_graphs_prev_k)
            latent_graphs_prev_k = latent_graphs_k.clone()

        return latent_graphs_k

    def _find_prior(self, batch):
        thresh = self._cfg_dict['prior_contact_thresh']
        w_class = torch.tensor([[thresh, -1.],[-thresh, 1.]], requires_grad=True).to(self._device)

        inp = torch.cat((
                torch.ones((batch.edge_attr.shape[0], 1), requires_grad=True).to(self._device),
                batch.edge_attr
            ), axis=1)

        out = inp@w_class.T
        # prior = torch.softmax(out, dim=1)

        flag = ((out[:,0]<0.).float()).repeat(out.shape[1],1).T

        slow_p = torch.softmax(out, dim=1)
        fast_p = torch.softmax(10.0*out, dim=1)
        prior = flag*slow_p + (1.-flag)*fast_p
        return prior

    def _evaluate_loss(self, batch):
        xtp1, *_ = self(batch)

        loss_fwd_dy = self._cfg_dict["weight_fwd_dy"]*F.mse_loss(xtp1, batch.y)

        loss_kld, loss_edge_pairs = 0., 0.
        
        if self._cfg_dict["loss_edge_prior"]:
            p_prior = self._find_prior(batch)
            loss_kld = self._edge_act * (torch.log(self._edge_act) - torch.log(p_prior))
            loss_kld = self._cfg_dict["weight_edge_prior"]*torch.mean(torch.sum(loss_kld, dim=1))

        if self._cfg_dict["loss_edge_pairs"]:
            loss_edge_pairs = self._cfg_dict["weight_edge_pairs"]*F.mse_loss(self._edge_act[::2,:], self._edge_act[1::2,:])
        
        return loss_fwd_dy, loss_kld, loss_edge_pairs

    def training_step(self, batch, batch_idx):

        loss_fwd_dy, loss_kld, loss_edge_pairs = self._evaluate_loss(batch)

        self.log("train_loss_fwd_dy", loss_fwd_dy)
        loss = loss_fwd_dy + 0.
        
        if self._cfg_dict["loss_edge_prior"]:
            self.log("train_loss_edge_prior", loss_kld)
            loss += loss_kld

        if self._cfg_dict["loss_edge_pairs"]:
            self.log("train_loss_edge_pairs", loss_edge_pairs)
            loss += loss_edge_pairs
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        loss_fwd_dy, loss_kld, loss_edge_pairs = self._evaluate_loss(batch)

        self.log("val_loss_fwd_dy", loss_fwd_dy)
        loss = loss_fwd_dy + 0.
        
        if self._cfg_dict["loss_edge_prior"]:
            self.log("val_loss_edge_prior", loss_kld)
            loss += loss_kld

        if self._cfg_dict["loss_edge_pairs"]:
            self.log("val_loss_edge_pairs", loss_edge_pairs)
            loss += loss_edge_pairs

        self.log("val_loss", loss)
