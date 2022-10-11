from omegaconf import OmegaConf
from isaacgym import gymapi

from Dynamic_GNN_structured_models.envs import *
from Dynamic_GNN_structured_models.skills import *
from tqdm import trange
import numpy as np
import os

def main():
    cfg = OmegaConf.load('cfg/collect_skill_data/franka_pickup_issaacgym_data.yaml')
    cfg_env = OmegaConf.load('cfg/envs/franka_pickup_isaacgym_env.yaml')

    if cfg['debug']:
        plot = True
        cfg_env['scene']['gui'] = 1
    else:
        plot = False
        cfg_env['scene']['gui'] = 0
        cfg_env['scene']['n_envs'] = 10
        cfg_env['scene']['es'] = 0

    env = eval(cfg['env'])(cfg_env)
    
    data_root_dir = cfg['data_root_dir']
    os.makedirs(data_root_dir, exist_ok=True)

    cfg_skill_dict = {k: v for k, v in cfg['skills'][cfg['skill_type']].items()}
    skill = eval(cfg['skill_type'])(cfg_skill_dict)

    expert_trajs = []
    for batch in trange(cfg_skill_dict['n_batches'], desc="Data collection"):
        data = skill.execute(env, env, T_exec_max=cfg_skill_dict['T_exec_max'], plot=plot, use_learned_model=False)
        expert_trajs.extend(data)
    
    if not cfg['debug']:
        expert_trajs = np.stack(expert_trajs, axis=0)
        np.savez(f'{data_root_dir}/franka_dynamic_sliding_pickup_small.npz', train_expert_trajs=expert_trajs)
        print("DATA SAVED")

if __name__ == "__main__":
    main()