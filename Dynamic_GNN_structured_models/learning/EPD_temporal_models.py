from Dynamic_GNN_structured_models.learning import *
from .model_utils import make_mlp
from .graph_network import *
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import numpy as np
from Dynamic_GNN_structured_models.datasets.data_utils import *

class EPDTemporalObsLinearModelwithModes(EPDLinearObsSpringMassDamperModelwithModes):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        super(EPDTemporalObsLinearModelwithModes, self).__init__(train_ds_gen=train_ds_gen, val_ds_gen=val_ds_gen, cfg_dict=cfg_dict)
        
    def _make_model(self):
        self._n_dyn_params_node = self._cfg_dict['dof']*5 if self._cfg_dict["learn_impact_params"] else self._cfg_dict['dof']*4

        if self._cfg_dict["heterogeneous_nodes_pn"]:
            self._processor_networks = torch.nn.ModuleList([GraphNetworkHeteroNodes({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])
        else:
            self._processor_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])

        self._inference_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                for i in range(self._cfg_dict['num_message_passing_steps_graphInf'])])

        self._fc_phi_R_act = torch.nn.GRU(
            input_size=self._cfg_dict_PN["D_R_out"],
            hidden_size=self._cfg_dict["num_edge_types"],
            batch_first=True).to(self._device)
        
        self._linear_model_parameters = make_mlp(
            self._cfg_dict['D_S'], self._cfg_dict['hidden_layers_mkc'] + [self._n_dyn_params_node], 
            prefix='linear_model_parameters_network', act='relu', last_act='sigmoid'
            ).to(self._device)

        
    def _process(self, processor_networks, latent_graphs_0, h0=None, T=None):
        # This implementation assumes that the number of edges/objects in each batch will be the same
        # i.e. it assumes that we will not be using Mixed datasets for this model

        start_at0 = True if h0 is None else False

        batch_size = latent_graphs_0.num_graphs

        latent_graphs_prev_k = latent_graphs_0.clone()
        for inference_network_k in self._inference_networks:
            latent_graphs_k = self._process_step(
                inference_network_k, latent_graphs_prev_k)
            latent_graphs_prev_k = latent_graphs_k.clone() 

        n_o = torch.sum(latent_graphs_0.edge_index[0,:]==0)+1 # number of edges coming from node 0 (gripper) equals number of objects TODO: remove hardcoded
        n_e = (n_o-1)*2 # TODO: remove hardcoded

        if start_at0:
            h0 = torch.zeros(1, batch_size, n_e, self._cfg_dict["num_edge_types"]).to(self._device)
        
        edge_attrs = latent_graphs_prev_k.edge_attr.view(batch_size, -1, self._D_R)

        self._edge_act_logits = []
        for indx in range(n_e):
            ht, _ = self._fc_phi_R_act(
                edge_attrs[:, indx::n_e, :],
                torch.zeros(1, batch_size, self._cfg_dict["num_edge_types"]).to(self._device)
                )
            self._edge_act_logits.append(ht.clone())

        self._edge_act_logits = torch.stack(self._edge_act_logits, dim=2)
        self._edge_act_logits = torch.flatten(self._edge_act_logits, start_dim=1, end_dim=2)
        self._edge_act_logits = self._edge_act_logits.view(-1, self._cfg_dict["num_edge_types"])
        
        if self._cfg_dict['sampling'] == 'argmax':
            self._edge_act = torch.softmax(self._edge_act_logits, dim=1)
            
            indx = self._edge_act.max(dim=1)[1].view(-1,1)
            edge_act_hard = torch.zeros_like(self._edge_act)
            edge_act_hard.scatter_(dim=1, index=indx, src=torch.ones_like(self._edge_act))
            self._edge_act_onehot = (edge_act_hard - self._edge_act).detach() + self._edge_act

        if self._cfg_dict['sampling'] == 'gumbel':
            self._edge_act_onehot = F.gumbel_softmax(self._edge_act_logits, tau=self.gumbel_tau, hard=True)
            self.gumbel_tau *= self._cfg_dict["gumbel_decay"]

        latent_graphs_prev_k = latent_graphs_0.clone()
        latent_graphs_prev_k.edge_act = self._edge_act_onehot.clone()
        for processor_network_k in processor_networks:
            latent_graphs_k = self._process_step(
                processor_network_k, latent_graphs_prev_k)
            latent_graphs_prev_k = latent_graphs_k.clone()

        return latent_graphs_prev_k
    
    def update_ht_fwd_prop(self, ht, zt):
        '''
        Inputs:
            ht: shape (1, batch_size, n_e, model._cfg_dict["num_edge_types"])
            zt: shape (batch_size*n_e, self._D_R)

        '''
        batch_size = zt.num_graphs
        edge_attrs = zt.edge_attr.view(batch_size, -1, self._D_R)
        n_e = ht.shape[2]
        ht_all = []
        for indx in range(n_e):
            _, _ht = self._fc_phi_R_act(
                edge_attrs[:, indx::n_e, :],
                ht[:,:,indx,:].contiguous()
                ) 
            ht_all.append(_ht.clone())
        return torch.stack(ht_all, dim=2)

class EPDTemporalObsLinearModelwithModesReactive(EPDLinearObsSpringMassDamperModelwithModes):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        super(EPDTemporalObsLinearModelwithModesReactive, self).__init__(train_ds_gen=train_ds_gen, val_ds_gen=val_ds_gen, cfg_dict=cfg_dict)
        
    def _make_model(self):
        self._n_dyn_params_node = self._cfg_dict['dof']*5 if self._cfg_dict["learn_impact_params"] else self._cfg_dict['dof']*4

        if self._cfg_dict["heterogeneous_nodes_pn"]:
            self._processor_networks = torch.nn.ModuleList([GraphNetworkHeteroNodes({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])
        else:
            self._processor_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                    for i in range(self._cfg_dict['num_message_passing_steps'])])

        self._inference_networks = torch.nn.ModuleList([GraphNetwork({**self._cfg_dict, **self._cfg_dict_PN}) 
                                                                for i in range(self._cfg_dict['num_message_passing_steps_graphInf'])])

        self._fc_phi_R_act = torch.nn.GRUCell(
            input_size=self._cfg_dict_PN["D_R_out"],
            hidden_size=self._cfg_dict["num_edge_types"]).to(self._device)
        
        self._linear_model_parameters = make_mlp(
            self._cfg_dict['D_S'], self._cfg_dict['hidden_layers_mkc'] + [self._n_dyn_params_node], 
            prefix='linear_model_parameters_network', act='relu', last_act='sigmoid'
            ).to(self._device)
    
    def GRU(self, edge_attr, h0):
        T = edge_attr.shape[1]
        batch_size = edge_attr.shape[0]
        
        ht = h0.clone()
        ht_all = []
        for i in range(T):
            ht = self.GRU_Cell(edge_attr[:, i, :], ht)
            ht_all.append(ht)
        return torch.stack(ht_all, dim=1)
    
    def GRU_Cell(self, edge_attr, ht=None):
        if ht == None:
            ht = torch.zeros(batch_size, self._cfg_dict["num_edge_types"]).to(self._device)
        
        _ht = self._fc_phi_R_act(edge_attr, ht)
        return torch.softmax(_ht, dim=1)
        
    def _process(self, processor_networks, latent_graphs_0, h0=None, T=None):
        
        if T is None:
            T = self.T
        
        start_at0 = True if h0 is None else False
            
        batch_size = latent_graphs_0.num_graphs

        latent_graphs_prev_k = latent_graphs_0.clone()
        for inference_network_k in self._inference_networks:
            latent_graphs_k = self._process_step(
                inference_network_k, latent_graphs_prev_k)
            latent_graphs_prev_k = latent_graphs_k.clone()

        n_o = torch.sum(latent_graphs_0.edge_index[0,:]==0)+1 # number of edges coming from node 0 (gripper) equals number of objects TODO: remove hardcoded
        n_e = (n_o-1)*2

        edge_attrs = latent_graphs_prev_k.edge_attr.view(batch_size, -1, self._D_R)

        self._edge_act = []

        if start_at0:
            h0 = torch.zeros(batch_size, n_e, self._cfg_dict["num_edge_types"]).to(self._device)
        else:
            h0 = h0.view(batch_size, n_e, self._cfg_dict["num_edge_types"])
        
        for indx in range(n_e):
            ht = self.GRU(
                    edge_attrs[:, indx::n_e, :], 
                    h0[:,indx,:]
                    )
            self._edge_act.append(ht.clone())

        self._edge_act = torch.stack(self._edge_act, dim=2)
        self._edge_act = torch.flatten(self._edge_act, start_dim=1, end_dim=2)
        self._edge_act = self._edge_act.view(-1, self._cfg_dict["num_edge_types"])

        if self._cfg_dict['sampling'] == 'argmax':
            indx = self._edge_act.max(dim=1)[1].view(-1,1)
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

    def update_ht_obs_contact(self, ht, zt, contacts):
        ht = self.GRU_Cell(zt.edge_attr, ht[0])
        for i in range(len(contacts)):
            if contacts[i]==1:
                ht[i*2] = ht[i*2]*0.99
                ht[i*2+1] = ht[i*2+1]*0.01
            else:
                ht[i*2] = ht[i*2]*0.01
                ht[i*2+1] = ht[i*2+1]*0.99
        return torch.softmax(ht, dim=1)[None,:,:]
    
    def update_ht_fwd_prop(self, ht, zt): # when used for LQR, n_e becomes batch_size here so we don't need to loop over n_e
        '''
        ht: shape (1, n_e, model._cfg_dict["num_edge_types"])
        '''
        return self.GRU_Cell(zt.edge_attr, ht[0])[None,:,:]

    def linearize_dynamics_obs_space(self, tau, h0, T):
        data_traj = self.ds.data_from_input_traj(tau)
        data = Batch.from_data_list([data_traj]).to(self._device)
        _, A, B, offset, _ = self._predict_next_state_via_linear_model(data, ht=h0, T=T)
        F, f = self.post_process_ABC(A, B, offset)
        return F, f


