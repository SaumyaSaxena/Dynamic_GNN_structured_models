from Dynamic_GNN_structured_models.envs import PickUp2DGymEnv, BlocksGraspXZ, DoorOpening
from Dynamic_GNN_structured_models.skills import *
from tqdm import trange
import numpy as np
import logging
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def main():
    batch_size = 500
    cfg_env = OmegaConf.load("../Dynamic_GNN_structured_models/cfg/envs/block_grasp_box2D_env.yaml")
    env = eval(cfg_env['env'])(cfg_env)

    skill_name = 'PickupPID'
    skill = eval(skill_name)()
    expert_trajs = []
    for batch in trange(batch_size, desc="Data collection"):
        data = skill.execute(env, env, T_exec_max=env.episode_len, plot=False, use_learned_model=False)
        if (not np.any(np.isnan(data))):
            expert_trajs.append(data)
    expert_trajs = np.stack(expert_trajs, axis=0)
    np.savez(f'/home/saumyas/Documents/experiment_data/Dynamic_GNN_structured_models/pick_up_2D_env_data/{skill_name}/data_pickup_2D_dt_0_05_T_200_zero_action_3objs.npz', 
                train_expert_trajs=expert_trajs)

if __name__ == "__main__":
    main()