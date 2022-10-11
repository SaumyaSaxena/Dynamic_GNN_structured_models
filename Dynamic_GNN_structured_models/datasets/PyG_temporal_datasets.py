import numpy as np
import torch
from torch_geometric.data import Dataset, Data, Batch
from Dynamic_GNN_structured_models.datasets.data_utils import MyData
from Dynamic_GNN_structured_models.datasets.data_utils import *

class AnalyticalEnvPickup1ObjTemporalDataset(Dataset):
    def __init__(self, normalize_data = True):
        super(AnalyticalEnvPickup1ObjTemporalDataset, self).__init__()

        data_raw = self.load_data()
        
        self.dataset_size = data_raw.shape[0]
        self.T = data_raw.shape[1]

        self.process_data(data_raw)

        self.dimensions = {
            "dataset_name" : 'AnalyticalEnvPickup1ObjTemporalDataset',
            "T": self.T,
            "N_O" : 2, # gripper and object
            "N_R" : self.edge_index.shape[1], # gripper-object and vice-versa
            "D_S" : 5, # joint pos (2), velocity (2), binary actuated/unactuated flag(1)
            "D_U" : 2, # dimension of control (ux, uy)
            "D_R" : 1, # edge params
            "D_G" : 2, # global params (zeros)
            "D_S_d" : 5, # joint pos (2), velocity (2), binary actuated/unactuated flag(1)
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.05,
            "dof" : 2,
            "prior_contact_thresh": 0.1
        }

    def load_data(self):
        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200.npz')
        data_raw0 = data_raw0['train_expert_trajs']

        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_RandomExploration.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data_raw2 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_zero_action.npz')
        data_raw2 = data_raw2['train_expert_trajs']

        data_raw = np.concatenate((data_raw0, data_raw1, data_raw2), axis=0)

        return data_raw

    def process_data(self, data_raw):
        self.observations = data_raw[:,:,:8,0].reshape(self.dataset_size, self.T*2, 4)
        self.actions = np.repeat(data_raw[:,:,8:,0], 2, axis=1)
        
        _data = Data(edge_index=torch.LongTensor(
                                    [[0,1],
                                    [1,0]]))
        _data.num_nodes = 2
        _batch = Batch.from_data_list([_data]*self.T)
        self.edge_index = _batch.edge_index.clone()
        self.edge_attrs = np.linalg.norm(self.observations[:,self.edge_index[0],:2] - self.observations[:,self.edge_index[1],:2], axis=2, keepdims=True) #norm of distance between the nodes

    def get_config(self):
        return self.dimensions
        
    def get(self, idx):
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        obs = np.append(self.observations[idx], np.tile([[1],[0]], (self.dimensions['T'],1)), axis=1 ) # shape = (self.dimensions['N_O']*self.dimensions['T'],self.dimensions['D_S']
        control = self.actions[idx] # shape = (self.dimensions['N_O']*self.dimensions['T'], self.dimensions['D_U'])
        edge_attr = self.edge_attrs[idx] # shape = (self.dimensions['N_R'], self.dimensions['D_R'])
        y = np.append(obs[self.dimensions['N_O']:, :], obs[-self.dimensions['N_O']:, :], axis=0)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=self.edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32)
            )
        return data
    
    def len(self):
        return self.dataset_size

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]
    
    def data_from_input(self, x, u, n=4):
        
        n_o = x.flatten().shape[0] // n
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32)
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)

        x = x.flatten().reshape(n_o, n)
        for i in range(n_o):
            obs[i, 0:4] = x[i, 0:4].copy()
            control[i,:] = u.squeeze()
        obs[0, 4] = 1 # Gripper actuated

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data
    
    def data_from_input_traj(self, tau):
        # tau has shape (T, n*N_O+m, 1), T = 200
        m=2
        n=4
        N_O = (tau.shape[1] - m) // n
        T = tau.shape[0]
        indicator = np.zeros((N_O,1))
        indicator[0,0] = 1
        obs = tau[:, :-m, 0].reshape(N_O*T, n)

        obs = np.append(obs, np.tile(indicator, (T,1)), axis=1 )
        control = np.repeat(tau[:, -m:, 0], N_O, axis=0)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        _data = Data(edge_index=torch.LongTensor(find_edge_index_pickup(N_O)))
        _data.num_nodes = N_O
        _batch = Batch.from_data_list([_data]*T)
        edge_index = _batch.edge_index.clone()

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data


class Box2DEnvPickup2ObjsTemporalDataset(Dataset):
    '''
    Non-mixed dataset: Contains trajectories with:
     - gripper picking up 2 objects 
     - gripper moving randomly
     - zero actions
    '''
    def __init__(self):
        super(Box2DEnvPickup2ObjsTemporalDataset, self).__init__()

        self.load_data()

        N_O = 3
        _data = Data(edge_index=torch.LongTensor(find_edge_index_pickup(N_O)))
        _data.num_nodes = N_O
        _batch = Batch.from_data_list([_data]*self.T)

        self.dimensions = {
            "dataset_name" : 'Box2DEnvPickup2ObjsTemporalDataset',
            "T": self.T,
            "N_O" : N_O, # gripper and object
            "N_R" : _batch.edge_index.shape[1], # gripper-object and vice-versa
            "D_S" : 5, # joint pos (2), velocity (2), binary actuated/unactuated flag(1)
            "D_U" : 2, # dimension of control (ux, uy)
            "D_R" : 1, # edge params
            "D_G" : 2, # global params (zeros)
            "D_S_d" : 5, # joint pos (2), velocity (2), binary actuated/unactuated flag(1)
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.05,
            "dof" : 2,
            "prior_contact_thresh": 0.2
        }
    
    def load_data(self):

        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickupPID/data_pickup2objects_2D_dt_0_05_T_200.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data_raw2 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_zero_action_3objs.npz')
        data_raw2 = data_raw2['train_expert_trajs']

        data_raw3 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_RandomExploration_3objs.npz')
        data_raw3 = data_raw3['train_expert_trajs']

        data = np.concatenate((data_raw1, data_raw2, data_raw3), axis=0)
        
        self.dataset_size = data.shape[0]
        self.T  = data.shape[1] - 1
        
        N_O=3
        n=4
        m=2
        pos = data[:,:-1,:-m,0].reshape(self.dataset_size, self.T *N_O, n)
        next_pos = data[:,1:,:-m,0].reshape(self.dataset_size, self.T *N_O, n)
        actions = np.repeat(data[:,:-1,-m:,0], N_O, axis=1)

        self.data = {
            'observations': pos,
            'actions': actions,
            'next_observations': next_pos
        }
        
    def get_config(self):
        return self.dimensions
        
    def get(self, idx):
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        
        obs = np.array(self.data['observations'][idx])
        N_O = obs.shape[0] // self.T
        indicator = np.zeros((N_O,1))
        indicator[0,0] = 1

        obs = np.append(obs, np.tile(indicator, (self.T,1)), axis=1 ) # shape = (self.dimensions['N_O']*self.dimensions['T'],self.dimensions['D_S']
        control = np.array(self.data['actions'][idx]) # shape = (self.dimensions['N_O']*self.dimensions['T'], self.dimensions['D_U'])
        
        _data = Data(edge_index=torch.LongTensor(find_edge_index_pickup(N_O)))
        _data.num_nodes = N_O
        _batch = Batch.from_data_list([_data]*self.T)

        edge_attr = np.linalg.norm(obs[_batch.edge_index[0],:2] - obs[_batch.edge_index[1],:2], axis=1, keepdims=True)
        y = np.append(np.array(self.data['next_observations'][idx]), np.tile(indicator, (self.T,1)), axis=1 )

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=_batch.edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32)
            )
        return data
    
    def len(self):
        return self.dataset_size

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]
    
    def data_from_input(self, x, u, n=4):
        
        n_o = x.flatten().shape[0] // n
        x = x.flatten().reshape(n_o, n)

        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)

        for i in range(n_o):
            obs[i, 0:4] = x[i, 0:4].copy()
            control[i,:] = u.squeeze()
        obs[0, 4] = 1 # Gripper actuated
        
        edge_index=torch.LongTensor(find_edge_index_pickup(n_o))

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data
    
    def data_from_input_traj(self, tau):
        # tau has shape (T, n*N_O+m, 1), T = 200
        m=2
        n=4
        N_O = (tau.shape[1] - m) // n
        T = tau.shape[0]
        indicator = np.zeros((N_O,1))
        indicator[0,0] = 1
        obs = tau[:, :-m, 0].reshape(N_O*T, n)

        obs = np.append(obs, np.tile(indicator, (T,1)), axis=1 )
        control = np.repeat(tau[:, -m:, 0], N_O, axis=0)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        _data = Data(edge_index=torch.LongTensor(find_edge_index_pickup(N_O)))
        _data.num_nodes = N_O
        _batch = Batch.from_data_list([_data]*T)
        edge_index = _batch.edge_index.clone()

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data

class Franka1ObjectPickupTemporalDataset(Dataset):
    def __init__(self, normalize_data = True):
        super(Franka1ObjectPickupTemporalDataset, self).__init__()
    
        self.data_raw = self.load_data()
        self._n_trajs = self.data_raw.shape[0]
        self.T = self.data_raw.shape[1] - 1
        self.dataset_size = self.data_raw.shape[0]

        self.process_data(self.data_raw)

        self.dimensions = {
            "dataset_name" : 'Franka1ObjectPickupTemporalDataset',
            "T": self.T,
            "N_O" : 2, # gripper and object
            "N_R" : self.edge_index.shape[1],
            "D_S" : 6, # pos (3), velocity (3)
            "D_G" : 2, # global params (zeros)
            "D_U" : 3, # dimension of control
            "D_R" : 1, # edge params
            "D_S_d" : 6, # pos (3), velocity (3),
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.01,
            "dof" : 3,
            "num_node_types": 2,
            "prior_contact_thresh": 0.019
        }
    
    def process_data(self, data_raw):
        m = 13
        
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': []
        }
        pos = np.concatenate((data_raw[:, :, 14:17],
                              data_raw[:, :, 21:24],
                              data_raw[:, :, 27:30],
                              data_raw[:, :, 34:37]), axis=2)
        self.data['observations'] = pos[:, :-1, :].reshape(self.dataset_size, self.T*2, 6)
        self.data['next_observations'] = pos[:, 1:, :].reshape(self.dataset_size, self.T*2, 6)
        self.data['actions'] = np.repeat(data_raw[:, :-1, -6:-3], 2, axis=1)

        _data = Data(edge_index=torch.LongTensor(
                                    [[0,1],
                                    [1,0]]))
        _data.num_nodes = 2
        _batch = Batch.from_data_list([_data]*self.T)
        self.edge_index = _batch.edge_index.clone()
        self.edge_attrs = np.linalg.norm(
            self.data['observations'][:,self.edge_index[0],:3] - self.data['observations'][:,self.edge_index[1],:3], 
            axis=2, keepdims=True) #norm of distance between the nodes
        self.node_type = np.tile(1.0*np.eye(2), (self.T, 1))

    def get_config(self):
        return self.dimensions

    def load_data(self):
        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_pickup_isaacgym_env_data/FrankaEEImpedanceControlDynamicPickUp/franka_dynamic_pickup1.npz')
        data_raw0 = data_raw0['train_expert_trajs']

        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_pickup_isaacgym_env_data/FrankaEEImpedanceControlDynamicPickUp/franka_dynamic_pickup2.npz')
        data_raw1 = data_raw1['train_expert_trajs']
        data = np.concatenate((data_raw0, data_raw1), axis=0)
        return data

    def get(self, idx):
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(self.data['observations'][idx], dtype=torch.float32),
            control=torch.tensor(self.data['actions'][idx], dtype=torch.float32),
            node_type=torch.tensor(self.node_type, dtype=torch.float32),
            edge_index=self.edge_index, 
            edge_attr=torch.tensor(self.edge_attrs[idx], dtype=torch.float32),
            y=torch.tensor(self.data['next_observations'][idx], dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data

    def data_from_input(self, x, u, n=None):
        # x, u are in cartesian space i.e. x.size = 6*N_O, u.size = 3
        n = self.dimensions['D_S']
        n_o = x.flatten().shape[0] // n
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32)
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

        edge_attr = np.linalg.norm(obs[edge_index[0],:self.dimensions['dof']] - obs[edge_index[1],:self.dimensions['dof']], axis=1, keepdims=True)

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
    
    def data_from_input_traj(self, tau):
        # tau has shape (T, n*N_O+m, 1), T = 200
        m=3
        n=6
        N_O = (tau.shape[1] - m) // n
        T = tau.shape[0]
        
        obs = tau[:, :-m, 0].reshape(N_O*T, n)
        control = np.repeat(tau[:, -m:, 0], N_O, axis=0)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        _data = Data(edge_index=torch.LongTensor(find_edge_index_pickup(N_O)))
        _data.num_nodes = N_O
        _batch = Batch.from_data_list([_data]*T)
        edge_index = _batch.edge_index.clone()

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)
        node_type = np.append(np.zeros((N_O,1)), np.ones((N_O,1)), axis=1)
        node_type[0,0] = 1.
        node_type[0,1] = 0.
        node_type = np.tile(node_type, (T, 1))

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            node_type=torch.tensor(node_type, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data

class RealFrankaCartesianSpace1ObjectPickupTemporalDataset(Franka1ObjectPickupTemporalDataset):
    def __init__(self, normalize_data = True):
        super(RealFrankaCartesianSpace1ObjectPickupTemporalDataset, self).__init__()
        self.dimensions['dataset_name'] = 'RealFrankaCartesianSpace1ObjectPickupTemporalDataset'
        self.dimensions['prior_contact_thresh'] = 0.01
        self.dimensions['dt'] = 0.001
    
    def load_data(self):
        data = []
        for i in range(1,19):
            data_raw = np.load(f'/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_real_world_data/PickupObject/data_franka_pickup_dt_0_001_T_10_{i}.npz')
            data.append(data_raw['train_expert_trajs'][:9977,:])
        data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/franka_real_world_data/PickupObject/data_franka_pickup_dt_0_001_T_10.npz')
        data.append(data_raw['train_expert_trajs'][:9977,:])

        data = np.stack(data, axis=0)
        return data
    
    def process_data(self, data_raw):
        m = 3
        
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': []
        }
        pos = data_raw[:, :, :-m]
        self.data['observations'] = pos[:, :-1, :].reshape(self.dataset_size, self.T*2, 6)
        self.data['next_observations'] = pos[:, 1:, :].reshape(self.dataset_size, self.T*2, 6)
        self.data['actions'] = np.repeat(data_raw[:, :-1, -m:], 2, axis=1)

        _data = Data(edge_index=torch.LongTensor(
                                    [[0,1],
                                    [1,0]]))
        _data.num_nodes = 2
        _batch = Batch.from_data_list([_data]*self.T)
        self.edge_index = _batch.edge_index.clone()
        self.edge_attrs = np.linalg.norm(
            self.data['observations'][:,self.edge_index[0],:3] - self.data['observations'][:,self.edge_index[1],:3], 
            axis=2, keepdims=True) #norm of distance between the nodes
        self.node_type = np.tile(1.0*np.eye(2), (self.T, 1))
    

class Box2DDoorOpening1DoorTemporalDataset(Franka1ObjectPickupTemporalDataset):
    def __init__(self, normalize_data = True):
        super(Box2DDoorOpening1DoorTemporalDataset, self).__init__()

        self.dimensions = {
            "dataset_name" : 'Box2DDoorOpening1DoorTemporalDataset',
            "T": self.T,
            "N_O" : 2, # gripper and door
            "N_R" : self.edge_index.shape[1],
            "D_S" : 4, # pos (3), velocity (3)
            "D_G" : 2, # global params (zeros)
            "D_U" : 2, # dimension of control
            "D_R" : 1, # edge params
            "D_S_d" : 4, # pos (3), velocity (3),
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.05,
            "dof" : 2,
            "num_node_types": 2,
            "prior_contact_thresh": 0.085
        }

    def load_data(self):
        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/door_opening_box2D_env_data/DoorOpeningPID/data_door_opening_2D_dt_0_05_T_100_0.npz')
        data_raw0 = data_raw0['train_expert_trajs']
        
        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/door_opening_box2D_env_data/DoorOpeningPID/data_door_opening_2D_dt_0_05_T_100_1.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data = np.concatenate((data_raw0, data_raw1), axis=0)
        return data

    def process_data(self, data_raw):
        m = 2
        
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'hinge_loc': []
        }
        pos = data_raw[:, :, :8].copy()
        hinge_loc = data_raw[:, :-1, 8:10].copy()
        self.data['hinge_loc'] = np.append(np.zeros(hinge_loc.shape), hinge_loc, axis=2).reshape(self.dataset_size, self.T*2, 2)

        self.data['observations'] = pos[:, :-1, :].reshape(self.dataset_size, self.T*2, 4)
        self.data['next_observations'] = pos[:, 1:, :].reshape(self.dataset_size, self.T*2, 4)
        self.data['actions'] = np.repeat(data_raw[:, :-1, -m:], 2, axis=1)

        _data = Data(edge_index=torch.LongTensor(
                                    [[0,1],
                                    [1,0]]))
        _data.num_nodes = 2
        _batch = Batch.from_data_list([_data]*self.T)
        self.edge_index = _batch.edge_index.clone()
        self.edge_attrs = np.linalg.norm(
            self.data['observations'][:,self.edge_index[0],:2] - self.data['observations'][:,self.edge_index[1],:2], 
            axis=2, keepdims=True) #norm of distance between the nodes
        self.node_type = np.tile(1.0*np.eye(2), (self.T, 1))

    def get(self, idx):
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(self.data['observations'][idx], dtype=torch.float32),
            hinge_loc=torch.tensor(self.data['hinge_loc'][idx], dtype=torch.float32),
            control=torch.tensor(self.data['actions'][idx], dtype=torch.float32),
            node_type=torch.tensor(self.node_type, dtype=torch.float32),
            edge_index=self.edge_index, 
            edge_attr=torch.tensor(self.edge_attrs[idx], dtype=torch.float32),
            y=torch.tensor(self.data['next_observations'][idx], dtype=torch.float32),
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data
    
    def data_from_input(self, x, u, n=None):
        if n is None:
            # gripper has size 4 and doors have size 6
            n_o = 1 + (x.flatten().shape[0] - 4) // 6
        else:
            # x, u are in cartesian space i.e. x.size = 4*N_O, u.size = 2
            n_o = x.flatten().shape[0] // 4
            
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32) # link params
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        hinge_loc = np.zeros(shape = (n_o,self.dimensions['dof']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        obs[0, :] = x[:4].copy()
        control[0,:] = u.flatten().copy()
        node_type[0, 0] = 1.
        for i in range(1, n_o):
            if n is None:
                obs[i, :] = x[4+6*(i-1):4+6*(i-1)+4]
                hinge_loc[i, :] = x[4+6*(i-1)+4:4+6*(i-1)+6]
            else:
                obs[i, :] = x[4+4*(i-1):4+4*(i-1)+4]
            control[i,:] = u.copy()
            node_type[i, 1] = 1.      

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)

        data = MyData(
            pos=torch.tensor(obs, dtype=torch.float32),
            hinge_loc=torch.tensor(hinge_loc, dtype=torch.float32),
            glob=torch.tensor(glob),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32), 
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y=torch.tensor(next_obs, dtype=torch.float32),
            y_global=torch.tensor(next_glob)
            )
        return data
    
    def data_from_input_traj(self, tau):
        # tau has shape (T, 4+6*n_doors+m, 1), T = 100
        m=2
        n=4
        dof = self.dimensions['dof']
        N_O = 1 + (tau.shape[1]-4-m) // 6
        T = tau.shape[0]
        pos_gripper = tau[:,:n]
        pos_doors_hinges = tau[:,n:-m].reshape(T*(N_O-1), 6)
        hinges = pos_doors_hinges[:, n:].reshape(T, dof*(N_O-1))
        hinge_loc = np.append(np.zeros((T,dof)),hinges,axis=1).reshape(T*(N_O), dof)
        
        pos_doors = pos_doors_hinges[:, :n].reshape(T, n*(N_O-1))
        obs = np.append(pos_gripper, pos_doors, axis=1).reshape(T*(N_O), n)

        control = np.repeat(tau[:, -m:], N_O, axis=0)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)

        _data = Data(edge_index=torch.LongTensor(find_edge_index_pickup(N_O)))
        _data.num_nodes = N_O
        _batch = Batch.from_data_list([_data]*T)
        edge_index = _batch.edge_index.clone()

        edge_attr = np.linalg.norm(obs[edge_index[0],:2] - obs[edge_index[1],:2], axis=1, keepdims=True)

        node_type = np.append(np.zeros((N_O,1)), np.ones((N_O,1)), axis=1)
        node_type[0,0] = 1.
        node_type[0,1] = 0.
        node_type = np.tile(node_type, (T, 1))

        data = MyData(
            glob=torch.tensor(glob, dtype=torch.float32), 
            pos=torch.tensor(obs, dtype=torch.float32),
            hinge_loc=torch.tensor(hinge_loc, dtype=torch.float32),
            control=torch.tensor(control, dtype=torch.float32),
            edge_index=edge_index, 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            node_type=torch.tensor(node_type, dtype=torch.float32), 
            y_global=torch.tensor(next_glob, dtype=torch.float32)
            )
        return data
    
if __name__ == "__main__":
    data = Box2DDoorOpening1DoorTemporalDataset()
    sample = data.sample()