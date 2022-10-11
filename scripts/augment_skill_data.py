from Dynamic_GNN_structured_models.envs import *
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf

def no_collision_with_traj(x_objs, env, traj): # TODO: generalize this function to any input traj
    n_objs_sampled = env._num_blocks - 1
    for i in range(n_objs_sampled):
        dist_g = np.linalg.norm(traj[:, :2, 0] - x_objs[i*env.n:i*env.n+2], axis=1) # distance from gripper
        dist_o = np.linalg.norm(traj[:, env.n:env.n+2, 0] - x_objs[i*env.n:i*env.n+2], axis=1) # distance from object
        if env._cfg['env_props']['dynamics']['blocks']['shape'] == 'circle':
            if np.any(dist_g < (env._radius_G+env._radius_B+env._collision_thresh)) or np.any(dist_o < (env._radius_B+env._radius_B+env._collision_thresh)):
                return False
        elif env._cfg['env_props']['dynamics']['blocks']['shape'] == 'box':
            if np.any(dist_g < (env._width_G+env._width_B)/np.sqrt(2)+env._collision_thresh) or np.any(dist_o < (env._width_B+env._width_B)/np.sqrt(2)+env._collision_thresh):
                return False
    return True

def augmented_traj(traj, env):
    T = traj.shape[0]
    keep_sampling = True
    while keep_sampling:
        x_objs = env.reset()[2*env.n:]
        if no_collision_with_traj(x_objs, env, traj):
            keep_sampling = False
    
    new_traj = np.concatenate((
                    traj[:, :-env.m, :], 
                    np.repeat(x_objs[None,:, None], T, axis=0), 
                    traj[:, -env.m:, :]), axis=1)
    return new_traj


def main():
    cfg_env = OmegaConf.load("../Dynamic_GNN_structured_models/cfg/envs/block_grasp_box2D_env.yaml")
    env = eval(cfg_env['env'])(cfg_env)
    
    data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200_slow.npz')
    data_raw = data_raw['train_expert_trajs']

    # data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_zero_action.npz')
    # data_raw = data_raw['train_expert_trajs']

    # data_raw = np.load('/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/RandomExploration/data_pickup_2D_dt_0_05_T_200_RandomExploration.npz')
    # data_raw = data_raw['train_expert_trajs']

    # data_raw = np.concatenate((data_raw0, data_raw1), axis=0)

    n_trajs = data_raw.shape[0]

    data_new = []
    for i in trange(n_trajs):
        data_new.append(augmented_traj(data_raw[i], env))
    data_new = np.stack(data_new, axis=0)
    
    np.savez(f'/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/PickToGoaliLQROpenLoop/data_pickup_2D_dt_0_05_T_200_slow_augmented3objs.npz', 
                train_expert_trajs=data_new)

if __name__ == "__main__":
    main()