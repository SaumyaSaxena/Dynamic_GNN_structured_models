# import d4rl_pybullet
import gym
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from Dynamic_GNN_structured_models.datasets.data_utils import *
import torch
        
class AnalyticalEnvPickup1ObjDataset(Dataset):
    def __init__(self, normalize_data = True):
        super(AnalyticalEnvPickup1ObjDataset, self).__init__()
        
        data_raw = self.load_data()

        self._n_trajs = data_raw.shape[0]
        self.T = data_raw.shape[1] - 1
        m = 2
        self.N_O = (data_raw.shape[2]-m) // 4
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': []
        }
        
        self.data['observations'] = data_raw[:, :-1, :-m, 0]
        self.data['actions'] = data_raw[:, :-1, -m:, 0]
        self.data['next_observations'] = data_raw[:, 1:, :-m, 0]

        self.data['observations'] = np.concatenate(self.data['observations'], axis=0)
        self.data['actions'] = np.concatenate(self.data['actions'], axis=0)
        self.data['next_observations'] = np.concatenate(self.data['next_observations'], axis=0)

        self.dataset_size = self.data['observations'].shape[0]

        self.edge_index = torch.LongTensor(self.find_edge_index_pickup(self.N_O))
        
        self.dimensions = {
            "dataset_name" : 'AnalyticalEnvPickup1ObjDataset',
            "T": 1,
            "N_O" : self.N_O, # gripper and object
            "N_R" : self.edge_index.shape[1], # gripper-object and vice-versa
            "D_S" : 5, # joint pos (2), velocity (2), binary actuated/unactuated flag(1) -- no action here
            "D_U" : 2, # dimension of control (ux, uy)
            "D_R" : 1, # edge params
            "D_G" : 2, # global params (zeros)
            "D_S_d" : 5, # joint pos (2), velocity (2), binary actuated/unactuated flag(1)
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.05,
            "dof" : 2,
            "prior_contact_thresh": 0.08
        }
    
    def find_edge_index_pickup(self, n):
        # n = 1(gripper) + num objects
        n_objects = n-1
        from_vec = np.zeros(n_objects)
        to_vec = np.arange(n_objects) + 1
        edge_index = np.stack([np.stack([from_vec, to_vec], axis=0), np.stack([to_vec, from_vec], axis=0)], axis=2).reshape(2,-1)
        return edge_index.astype(int)

    def get_config(self):
        return self.dimensions

    def load_data(self):

        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200.npz')
        data_raw0 = data_raw0['train_expert_trajs']

        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_RandomExploration.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data_raw2 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_zero_action.npz')
        data_raw2 = data_raw2['train_expert_trajs']

        return np.concatenate((data_raw0, data_raw1, data_raw2), axis=0)

    def get(self, idx):
        x = np.array(self.data['observations'][idx])
        u = np.array(self.data['actions'][idx])
        next_x = np.array(self.data['next_observations'][idx])

        n_o = x.flatten().shape[0] // 4
        edge_index = self.find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32)
        
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)

        for i in range(n_o):
            obs[i, 0:4] = x[i*4:(i+1)*4]
            next_obs[i, 0:4] = next_x[i*4:(i+1)*4]
            control[i,:] = u.copy()
        obs[0, 4] = 1 # Gripper actuated
        next_obs[0, 4] = 1 # Gripper actuated

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

    def len(self):
        return self.dataset_size

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]

    def data_from_input(self, x, u, n=4):
        n_o = x.flatten().shape[0] // n
        edge_index = self.find_edge_index_pickup(n_o)
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


class AnalyticalEnvPickup1Obj1DistractorDataset(AnalyticalEnvPickup1ObjDataset):
    def __init__(self, normalize_data = True):
        super(AnalyticalEnvPickup1Obj1DistractorDataset, self).__init__()
        self.dimensions['dataset_name'] = 'AnalyticalEnvPickup1Obj1DistractorDataset'
    
    def load_data(self):
        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200_big_augmented3objs.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data_raw2 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_zero_action_augmented3objs.npz')
        data_raw2 = data_raw2['train_expert_trajs']

        data_raw3 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_RandomExploration_augmented3objs.npz')
        data_raw3 = data_raw3['train_expert_trajs']

        data = np.concatenate((data_raw1, data_raw2, data_raw3), axis=0)

        return data

class Box2DEnvPickup2ObjsDataset(AnalyticalEnvPickup1Obj1DistractorDataset):
    def __init__(self, normalize_data = True):
        super(Box2DEnvPickup2ObjsDataset, self).__init__()
        self.dimensions['dataset_name'] = 'Box2DEnvPickup2ObjsDataset'
        self.dimensions['prior_contact_thresh'] = 0.17
    
    def load_data(self):
        data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickupPID/data_pickup2objects_2D_dt_0_05_T_100.npz')
        data_raw = data_raw['train_expert_trajs']

        return data_raw

class AnalyticalEnvPickup1Obj1DistractorDatasetMixed(AnalyticalEnvPickup1Obj1DistractorDataset):
    def __init__(self, normalize_data = True):
        super(AnalyticalEnvPickup1Obj1DistractorDatasetMixed, self).__init__()
        self.dimensions['dataset_name'] = 'AnalyticalEnvPickup1Obj1DistractorDatasetMixed'
        obs = self.data['observations'].copy()
        actions = self.data['actions'].copy()
        next_obs = self.data['next_observations'].copy()
        n=4
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': []
        }
        for i in range(2, self.N_O+1):
            self.data['observations'].extend(obs[:,:n*i].tolist())
            self.data['actions'].extend(actions.tolist())
            self.data['next_observations'].extend(next_obs[:,:n*i].tolist())

        self.dataset_size = len(self.data['observations'])


class Box2DEnvPickup1Obj1DistractorPickup2ObjsDatasetMixed(AnalyticalEnvPickup1Obj1DistractorDatasetMixed):
    def __init__(self, normalize_data = True):
        super(Box2DEnvPickup1Obj1DistractorPickup2ObjsDatasetMixed, self).__init__()
        self.dimensions['dataset_name'] = 'Box2DEnvPickup1Obj1DistractorPickup2ObjsDatasetMixed'
        self.dimensions['prior_contact_thresh'] = 0.17

        data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickupPID/data_pickup2objects_2D_dt_0_05_T_100.npz')
        data_raw = data_raw['train_expert_trajs']

        m=2
        obs = data_raw[:, :-1, :-m, 0]
        actions = data_raw[:, :-1, -m:, 0]
        next_obs = data_raw[:, 1:, :-m, 0]

        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        next_obs = np.concatenate(next_obs, axis=0)

        self.data['observations'].extend(obs.tolist())
        self.data['actions'].extend(actions.tolist())
        self.data['next_observations'].extend(next_obs.tolist())
        
        self.dataset_size = len(self.data['observations'])

    def load_data(self):
        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200_big_augmented3objs.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200_slow_augmented3objs.npz')
        data_raw0 = data_raw0['train_expert_trajs']

        data = np.concatenate((data_raw0, data_raw1), axis=0)
        return data


class Box2DEnvDoorOpening1DoorDataset(Dataset):
    def __init__(self, normalize_data = True):
        super(Box2DEnvDoorOpening1DoorDataset, self).__init__()
        
        data_raw = self.load_data()
        self._n_trajs = data_raw.shape[0]
        self.T = data_raw.shape[1] - 1

        self.dimensions = {
            "dataset_name" : 'Box2DEnvDoorOpening1DoorDataset',
            "T": 1,
            "N_O" : 2, # gripper and object
            "N_R" : 2,
            "D_S" : 4, # pos (2), velocity (2)
            "D_G" : 2, # global params (zeros)
            "D_U" : 2, # dimension of control
            "D_R" : 1, # edge params
            "D_S_d" : 4, # pos (2), velocity (2),
            "D_G_d" : 2, # global params (zeros)
            "dt" : 0.05,
            "dof" : 2,
            "num_node_types": 2,
            "prior_contact_thresh": 0.085
        }
        self.process_data(data_raw)
    
    def load_data(self):
        data_raw0 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/door_opening_box2D_env_data/DoorOpeningPID/data_door_opening_2D_dt_0_05_T_100_big.npz')
        data_raw0 = data_raw0['train_expert_trajs']

        data_raw1 = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/door_opening_box2D_env_data/DoorOpeningPID/data_door_opening_2D_dt_0_05_T_100_slow.npz')
        data_raw1 = data_raw1['train_expert_trajs']

        data = np.concatenate((data_raw0, data_raw1), axis=0)
        return data
    
    def process_data(self, data_raw):
        m = 2
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

    def get(self, idx):
        x = np.array(self.data['observations'][idx])
        u = np.array(self.data['actions'][idx])
        next_x = np.array(self.data['next_observations'][idx])

        n_o = 1 + (x.flatten().shape[0] - 4) // 6
        
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32)
        obs = np.zeros(shape = (n_o,self.dimensions['D_S']), dtype = np.float32)
        hinge_loc = np.zeros(shape = (n_o,self.dimensions['dof']), dtype = np.float32)
        next_obs = np.zeros(shape = (n_o,self.dimensions['D_S_d']), dtype = np.float32)
        control = np.zeros(shape = (n_o,self.dimensions['D_U']), dtype = np.float32)
        glob = np.zeros(shape = (self.dimensions['D_G'],), dtype = np.float32)
        next_glob = np.zeros(shape = (self.dimensions['D_G_d'],), dtype = np.float32)
        node_type = np.zeros(shape = (n_o,self.dimensions['num_node_types']), dtype = np.float32)

        obs[0, :] = x[:4].copy()
        next_obs[0, :] = next_x[:4].copy()
        control[0,:] = u.copy()
        node_type[0, 0] = 1.
        for i in range(1, n_o):
            obs[i, :] = x[4+6*(i-1):4+6*(i-1)+4]
            next_obs[i, :] = next_x[4+6*(i-1):4+6*(i-1)+4]
            control[i,:] = u.copy()
            node_type[i, 1] = 1. 
            hinge_loc[i, :] = x[4+6*(i-1)+4:4+6*(i-1)+6]
        
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

    def data_from_input(self, x, u, n=None):
        if n is None:
            # gripper has size 4 and doors have size 6
            n_o = 1 + (x.flatten().shape[0] - 4) // 6
        else:
            # x, u are in cartesian space i.e. x.size = 4*N_O, u.size = 2
            n_o = x.flatten().shape[0] // 4
            
        edge_index = find_edge_index_pickup(n_o)
        n_r = edge_index.shape[1]

        edge_attr = np.ones(shape = (n_r, self.dimensions['D_R']), dtype = np.float32)
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

    def get_config(self):
        return self.dimensions

    def len(self):
        return self.dataset_size

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]

if __name__ == "__main__":
    data = Box2DEnvDoorOpening1DoorDataset()
    sample = data.sample()