import numpy as np
from torch_geometric.data import Data

class MyData(Data):
    def __cat_dim__(self, key, item):
        if key == 'glob' or key == 'y_global':
            return None
        else:
            return super().__cat_dim__(key, item)

def find_edge_index_pickup(n):
    # n = 1(gripper) + num objects
    n_objects = n-1
    from_vec = np.zeros(n_objects)
    to_vec = np.arange(n_objects) + 1
    # edge_index = np.append( np.stack([from_vec, to_vec], axis=0), np.stack([to_vec, from_vec], axis=0), axis=1)
    edge_index = np.stack([np.stack([from_vec, to_vec], axis=0), np.stack([to_vec, from_vec], axis=0)], axis=2).reshape(2,-1)
    return edge_index.astype(int)