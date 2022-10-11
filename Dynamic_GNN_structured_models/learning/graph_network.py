import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter
from .model_utils import make_mlp
import torch.nn.functional as F

class GraphNetwork(MessagePassing):
    def __init__(self, cfg_dict=None):
        super(GraphNetwork, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.save_hyperparameters(cfg_dict)
        gpu_load = cfg_dict.get('gpu', 0)
        self._device = f'cuda:{gpu_load}'

        self._N_O = cfg_dict["N_O"]

        self._D_S_in = cfg_dict["D_S_in"]
        self._D_R_in = cfg_dict["D_R_in"]
        self._D_G_in = cfg_dict["D_G_in"]

        self._D_S_out = cfg_dict["D_S_out"]
        self._D_R_out = cfg_dict["D_R_out"]
        self._D_G_out = cfg_dict["D_G_out"]       

        self.hidden_layers_phi_R = cfg_dict["hidden_layers_phi_R"]
        self.hidden_layers_phi_O = cfg_dict["hidden_layers_phi_O"]

        self._cfg_dict = cfg_dict

        self._make_model()

    def _make_model(self):

        # MLP for edges - edge attribute update
        self._fc_phi_R = make_mlp(
                            2 * self._D_S_in + self._D_R_in, self.hidden_layers_phi_R + [self._D_R_out],
                            prefix='fc_phi_R', last_act='relu'
                        ).to(self._device)
        # MLP for each object - node attribute update
        self._fc_phi_O = make_mlp(
                            self._D_S_in + self._D_R_out, self.hidden_layers_phi_O + [self._D_S_out],
                            prefix='fc_phi_O_mu', last_act='relu'
                        ).to(self._device)

    def forward(self, data):
        self._edge_attribute_out = None
        try:
            data.edge_act
        except:
            data.edge_act = None

        pos_out = self.propagate(
            glob=data.glob, pos=data.pos, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr,
            edge_act = data.edge_act
            )
        data = data.clone()
        data.pos = pos_out.clone()
        data.edge_attr = self._edge_attribute_out.clone()
        return data

    def message(self, pos_i, pos_j, edge_attr, edge_act):
        x_ij = torch.cat([pos_i, pos_j, edge_attr], dim=1) # TODO: Add global feature here?
        out = self._fc_phi_R(x_ij)
        if edge_act is not None:
            self._edge_attribute_out = out*edge_act[:,0,None].repeat(1, out.shape[1])
        else:
            self._edge_attribute_out = out.clone()
        return self._edge_attribute_out

    def update(self, node_ij_aggr, glob, pos):
        inp = torch.cat([node_ij_aggr, pos], dim=1)
        return self._fc_phi_O(inp)

class GraphNetworkHeteroNodes(GraphNetwork):
    def __init__(self, cfg_dict=None):
        super(GraphNetworkHeteroNodes, self).__init__(cfg_dict=cfg_dict)

    def _make_model(self):

        # MLP for edges - edge attribute update
        self._fc_phi_R = make_mlp(
                            2 * self._D_S_in + self._D_R_in, self.hidden_layers_phi_R + [self._D_R_out],
                            prefix='fc_phi_R', last_act='relu'
                        ).to(self._device)
        
        # MLP for each object - node attribute update
        self._fc_phi_O = torch.nn.ModuleList()
        for _ in range(self._cfg_dict["num_node_types"]):
            module = make_mlp(
                                self._D_S_in + self._D_R_out, self.hidden_layers_phi_O + [self._D_S_out],
                                prefix='fc_phi_O_mu', last_act='relu'
                            ).to(self._device)
            self._fc_phi_O.append(module)
        
    def forward(self, data):
        self._edge_attribute_out = None
        try:
            data.edge_act
        except:
            data.edge_act = None

        pos_out = self.propagate(
            glob=data.glob, pos=data.pos, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr,
            edge_act = data.edge_act,
            node_type = data.node_type
            )
        data = data.clone()
        data.pos = pos_out.clone()
        data.edge_attr = self._edge_attribute_out.clone()
        return data

    def update(self, node_ij_aggr, glob, pos, node_type):
        inp = torch.cat([node_ij_aggr, pos], dim=1)
        out = 0.
        for indx in range(self._cfg_dict["num_node_types"]):
            node_out = self._fc_phi_O[indx](inp)
            out += node_type[:,indx,None]*node_out
        return out


class GraphNetworkModes(GraphNetwork):
    def __init__(self, cfg_dict=None):
        super(GraphNetworkModes, self).__init__(cfg_dict=cfg_dict)

    def _make_model(self):

        # MLP for edges - edge attribute update, one for each edge type
        self._fc_phi_R = torch.nn.ModuleList()
        for _ in range(self._cfg_dict["num_edge_types"]):
            module = make_mlp(
                                2 * self._D_S_in + self._D_R_in, self.hidden_layers_phi_R + [self._D_R_out],
                                prefix='fc_phi_R', last_act='relu'
                            ).to(self._device)
            self._fc_phi_R.append(module)

        # MLP for each object - node attribute update
        self._fc_phi_O = make_mlp(
                            self._D_S_in + self._D_R_out, self.hidden_layers_phi_O + [self._D_S_out],
                            prefix='fc_phi_O_mu', last_act='relu'
                        ).to(self._device)

    def message(self, pos_i, pos_j, edge_attr, edge_act):
        x_ij = torch.cat([pos_i, pos_j, edge_attr], dim=1)
        self._edge_attribute_out = 0.
        for indx in range(self._cfg_dict["num_edge_types"]):
            edge_attr_out = self._fc_phi_R[indx](x_ij)
            self._edge_attribute_out += edge_act[:,indx,None]*edge_attr_out
        return self._edge_attribute_out