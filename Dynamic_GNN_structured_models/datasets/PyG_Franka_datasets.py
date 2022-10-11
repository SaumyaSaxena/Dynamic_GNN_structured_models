import gym
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from Dynamic_GNN_structured_models.datasets.data_utils import *
import torch

class FrankaCartesianSpace1ObjectPickupDataset(Dataset):
    def __init__(self, normalize_data = True):
        super(FrankaCartesianSpace1ObjectPickupDataset, self).__init__()
        
        self.data_raw = self.load_data()
        self._n_trajs = self.data_raw.shape[0]
        self.T = self.data_raw.shape[1] - 1

        self.dimensions = {
            "dataset_name" : 'FrankaCartesianSpace1ObjectPickupDataset',
            "T": 1,
            "N_O" : 2, # gripper and object
            "N_R" : 1,
            "D_S" : 6, # pos (3), velocity (3)
            "D_G" : 2, # global params (zeros)
            "D_U" : 3, # dimension of control
            "D_R" : 1, # edge params
            "D_S_d" : 6, # pos (3), velocity (3),
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.01,
            "dof" : 3,
            "num_node_types": 2,
            "prior_contact_thresh": 0.020
        }
        
        self.process_data(self.data_raw)
        
    def process_data(self, data_raw):
        m = 13
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': []
        }
        self.data['observations'] = data_raw[:, :-1, :-m]
        self.data['actions'] = data_raw[:, :-1, -m:]
        self.data['next_observations'] = data_raw[:, 1:, :-m]

        self.data['observations'] = np.concatenate(self.data['observations'], axis=0)
        self.data['actions'] = np.concatenate(self.data['actions'], axis=0)
        self.data['next_observations'] = np.concatenate(self.data['next_observations'], axis=0)

        self.dataset_size = self.data['observations'].shape[0]

    def get_config(self):
        return self.dimensions

    def load_data(self):
        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_pickup_isaacgym_env_data/FrankaEEImpedanceControlDynamicPickUp/franka_dynamic_pickup1.npz')
        data_raw0 = data_raw0['train_expert_trajs']

        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_pickup_isaacgym_env_data/FrankaEEImpedanceControlDynamicPickUp/franka_dynamic_pickup2.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        # data = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_pickup_isaacgym_env_data/FrankaEEImpedanceControlDynamicPickUp/franka_dynamic_pickup_small.npz')
        # data = data['train_expert_trajs']

        data = np.concatenate((data_raw0, data_raw1), axis=0)
        
        return data

    def get(self, idx):
        x = np.array(self.data['observations'][idx])
        u = np.array(self.data['actions'][idx])
        next_x = np.array(self.data['next_observations'][idx])

        n_o = 1 + (x.flatten().shape[0] - 27) // 13
        
        if n_o == 1:
            edge_index = np.array([[0],[0]])
        else:
            edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        obs[0, :] = np.append(x[14:17], x[21:24])
        next_obs[0, :] = np.append(next_x[14:17], next_x[21:24])
        control[0,:] = u[-6:-3]
        node_type[0, 0] = 1.
        for i in range(1, n_o):
            obs[i, :] = np.append(x[27+13*(i-1):27+13*(i-1)+3], x[27+13*(i-1)+7:27+13*(i-1)+10])
            next_obs[i, :] = np.append(next_x[27+13*(i-1):27+13*(i-1)+3], next_x[27+13*(i-1)+7:27+13*(i-1)+10]) #- self.data['observations'][idx, i*4:(i+1)*4]
            control[i,:] = u[-6:-3]
            node_type[i, 1] = 1. 
        
        if n_o > 1:
            edge_attr = np.linalg.norm(obs[edge_index[0],:3] - obs[edge_index[1],:3], axis=1, keepdims=True)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data

    def data_from_input(self, x, u, n=6):
        # x, u are in cartesian space i.e. x.size = 6*N_O, u.size = 3
        n_o = x.flatten().shape[0] // n
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        x = x.flatten().reshape(n_o, n)
        node_type[0, 0] = 1.
        for i in range(n_o):
            obs[i, :] = x[i, :].copy()
            control[i,:] = u.flatten().copy()
            if i>0:
                node_type[i, 1] = 1.

        edge_attr = np.linalg.norm(obs[edge_index[0],:3] - obs[edge_index[1],:3], axis=1, keepdims=True)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data

    def len(self):
        return self.dataset_size

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]

class FrankaCartesianSpace1ObjectPickupSlidingDataset(FrankaCartesianSpace1ObjectPickupDataset):
    def __init__(self, normalize_data = True):
        super(FrankaCartesianSpace1ObjectPickupSlidingDataset, self).__init__()
        self.dimensions['N_O'] = 3
        self.dimensions['num_node_types'] = 3
        self.dimensions['dataset_name'] = 'FrankaCartesianSpace1ObjectPickupSlidingDataset'
        self.dimensions['prior_contact_thresh'] = 0.021
    
    def get(self, idx):
        x = np.array(self.data['observations'][idx])
        u = np.array(self.data['actions'][idx])
        next_x = np.array(self.data['next_observations'][idx])

        n_o = 1 + (x.flatten().shape[0] - 27) // 13
        n_obj = n_o - 1

        n_o += 1 # Table added
        
        if n_o == 1:
            edge_index = np.array([[0],[0]])
        else:
            edge_index = self.find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        obs[0, :] = np.append(x[14:17], x[21:24])
        next_obs[0, :] = np.append(next_x[14:17], next_x[21:24])
        control[0,:] = u[-6:-3]
        node_type[0, 0] = 1.
        for i in range(1, n_o-1):
            obs[i, :] = np.append(x[27+13*(i-1):27+13*(i-1)+3], x[27+13*(i-1)+7:27+13*(i-1)+10])
            next_obs[i, :] = np.append(next_x[27+13*(i-1):27+13*(i-1)+3], next_x[27+13*(i-1)+7:27+13*(i-1)+10]) #- self.data['observations'][idx, i*4:(i+1)*4]
            control[i,:] = u[-6:-3]
            node_type[i, 1] = 1. 
        
        # Adding door as last node
        # obs[n_o-1,:2] = obs[1, :2].copy()
        obs[n_o-1,2] = 0.5 # table height
        obs[n_o-1,3:] = np.zeros(3)
        control[n_o-1,:] = u[-6:-3]
        node_type[n_o-1, 2] = 1. 

        next_obs[n_o-1, :] = next_obs[1,:].copy()

        if n_o > 1:
            # dist between gripper and objects
            edge_attr_g_o = np.linalg.norm(obs[edge_index[0,:-2*n_obj],:3] - obs[edge_index[1,:-2*n_obj],:3], axis=1, keepdims=True)
            # dist between table and objects
            edge_attr_t_o = np.linalg.norm(obs[edge_index[0,-2*n_obj:],2, None] - obs[edge_index[1,-2*n_obj:],2, None], axis=1, keepdims=True) - 0.039
            edge_attr = np.append(edge_attr_g_o, edge_attr_t_o, axis=0)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data

    def data_from_input(self, x, u, n=6):
        # x, u are in cartesian space i.e. x.size = 6*N_O, u.size = 3
        n_o = x.flatten().shape[0] // n
        n_obj = n_o - 1
        n_o += 1 # Table added
        edge_index = self.find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        x = x.flatten().reshape(n_o, n)
        node_type[0, 0] = 1.
        for i in range(n_o-1):
            obs[i, :] = x[i, :].copy()
            control[i,:] = u.flatten().copy()
            if i>0:
                node_type[i, 1] = 1.

        # obs[n_o-1,:2] = obs[1, :2].copy()
        obs[n_o-1,2] = 0.5 # table height
        obs[n_o-1,3:] = np.zeros(3)
        control[n_o-1,:] = u[-6:-3]
        node_type[n_o-1, 2] = 1. 

        # dist between gripper and objects
        edge_attr_g_o = np.linalg.norm(obs[edge_index[0,:-2*n_obj],:3] - obs[edge_index[1,:-2*n_obj],:3], axis=1, keepdims=True)
        # dist between table and objects
        edge_attr_t_o = np.linalg.norm(obs[edge_index[0,-2*n_obj:],2, None] - obs[edge_index[1,-2*n_obj:],2, None], axis=1, keepdims=True)
        edge_attr = np.append(edge_attr_g_o, edge_attr_t_o, axis=0)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data

    def load_data(self):
        data = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_pickup_isaacgym_env_data/FrankaEEImpedanceControlDynamicSlidePickUp/franka_dynamic_sliding_pickup.npz')
        data = data['train_expert_trajs']
        return data

    def find_edge_index_pickup(self, n):
        # n = 1(gripper) + num objects
        n_objects = n-2 # table is the last node
        from_vec = np.zeros(n_objects) # gripper node=0
        to_vec = np.arange(n_objects) + 1
        # edge_index = np.append( np.stack([from_vec, to_vec], axis=0), np.stack([to_vec, from_vec], axis=0), axis=1)
        edge_index_g_o = np.stack([np.stack([from_vec, to_vec], axis=0), np.stack([to_vec, from_vec], axis=0)], axis=2).reshape(2,-1)
        
        # from table to objects
        from_vec = np.zeros(n_objects) + n-1 # table node=n-1
        edge_index_t_o = np.stack([np.stack([from_vec, to_vec], axis=0), np.stack([to_vec, from_vec], axis=0)], axis=2).reshape(2,-1)
        
        edge_index = np.append(edge_index_g_o, edge_index_t_o, axis=1)
        return edge_index.astype(int)

class RealFrankaCartesianSpace1ObjectPickupDataset(Dataset):
    def __init__(self, normalize_data = True):
        super(RealFrankaCartesianSpace1ObjectPickupDataset, self).__init__()
        
        self.data_raw = self.load_data()
        self._n_trajs = self.data_raw.shape[0]
        self.T = self.data_raw.shape[1] - 1

        self.dimensions = {
            "dataset_name" : 'RealFrankaCartesianSpace1ObjectPickupDataset',
            "T": 1,
            "N_O" : 2, # gripper and object
            "N_R" : 1,
            "D_S" : 6, # pos (3), velocity (3)
            "D_G" : 2, # global params (zeros)
            "D_U" : 3, # dimension of control
            "D_R" : 1, # edge params
            "D_S_d" : 6, # pos (3), velocity (3),
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.001,
            "dof" : 3,
            "num_node_types": 2,
            "prior_contact_thresh": 0.01
        }
        
        self.process_data(self.data_raw)
        
    def process_data(self, data_raw):
        m = 3
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': []
        }
        self.data['observations'] = data_raw[:, :-1, :-m]
        self.data['actions'] = data_raw[:, :-1, -m:]
        self.data['next_observations'] = data_raw[:, 1:, :-m]

        self.data['observations'] = np.concatenate(self.data['observations'], axis=0)
        self.data['actions'] = np.concatenate(self.data['actions'], axis=0)
        self.data['next_observations'] = np.concatenate(self.data['next_observations'], axis=0)

        self.dataset_size = self.data['observations'].shape[0]

    def get_config(self):
        return self.dimensions

    def load_data(self):
        data = []
        for i in range(1,19):
            data_raw = np.load(f'/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_real_world_data/PickupObject/data_franka_pickup_dt_0_001_T_10_{i}.npz')
            data.append(data_raw['train_expert_trajs'][:9977,:])
        data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_real_world_data/PickupObject/data_franka_pickup_dt_0_001_T_10.npz')
        data.append(data_raw['train_expert_trajs'][:9977,:])

        data = np.stack(data, axis=0)
        return data

    def get(self, idx):
        x = np.array(self.data['observations'][idx])
        u = np.array(self.data['actions'][idx])
        next_x = np.array(self.data['next_observations'][idx])

        n_o = x.flatten().shape[0]//6
        
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        obs = x.flatten().reshape(n_o, self.dimensions['D_S'])
        next_obs = next_x.flatten().reshape(n_o, self.dimensions['D_S'])
        control[0,:] = u.flatten().copy()
        control[1,:] = u.flatten().copy()
        node_type[0, 0] = 1.
        node_type[1, 1] = 1.
        
        edge_attr = np.linalg.norm(obs[edge_index[0],:3] - obs[edge_index[1],:3], axis=1, keepdims=True)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data

    def data_from_input(self, x, u, n=6):
        # x, u are in cartesian space i.e. x.size = 6*N_O, u.size = 3
        n_o = x.flatten().shape[0] // n
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        x = x.flatten().reshape(n_o, n)
        node_type[0, 0] = 1.
        for i in range(n_o):
            obs[i, :] = x[i, :].copy()
            control[i,:] = u.flatten().copy()
            if i>0:
                node_type[i, 1] = 1.

        edge_attr = np.linalg.norm(obs[edge_index[0],:3] - obs[edge_index[1],:3], axis=1, keepdims=True)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data

    def len(self):
        return self.dataset_size

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]


if __name__ == "__main__":
    data = FrankaCartesianSpace1ObjectPickupDataset()
    sample = data.sample()