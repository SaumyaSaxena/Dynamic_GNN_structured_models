from Dynamic_GNN_structured_models.envs import *
from Dynamic_GNN_structured_models.skills import *
from Dynamic_GNN_structured_models.learning import *
from Dynamic_GNN_structured_models.datasets import *
import torch
from tqdm import trange
import numpy as np
import logging
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def main():
    batch_size = 1

    cfg_env = OmegaConf.load("../Dynamic_GNN_structured_models/cfg/envs/block_grasp_box2D_env.yaml")
    env = eval(cfg_env['env'])(cfg_env)

    if cfg_env['env'] == 'FrankaPickupIsaacgymEnv':
        cfg_skill = OmegaConf.load('cfg/collect_skill_data/franka_pickup_issaacgym_data.yaml')
        cfg_skill_dict = {k: v for k, v in cfg_skill['skills'][cfg_skill['skill_type']].items()}
        skill = eval(cfg_skill['skill_type'])(cfg_skill_dict)
    else:
        skill = PickToGoaliLQROpenLoopReactive()

    model_path = './outputs/2022-09-03/11-18-14/EPD_temporal_forward_model_analytical_2objects_modes/'
    checkpoint = torch.load(model_path + 'checkpoints/last.ckpt', map_location='cuda:0')
    cfg_dict = OmegaConf.load(model_path + 'logs/default/version_0/hparams.yaml')
    
    model_name = cfg_dict['model_name']
    dataset_name = cfg_dict['dataset_name']
    
    dataset = eval(dataset_name)()
    train_size = int(len(dataset) * cfg_dict['train_val_split'])
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, (train_size, val_size))

    cfg_dict['N_O'] = env.N_O # needed for generalization of model to more nodes
    cfg_dict['gpu'] = 0 # to enable testing on machines with 1 gpu
    model = eval(model_name)(
            train_ds_gen=lambda : dataset, 
            val_ds_gen=lambda: dataset, 
            cfg_dict=cfg_dict
            )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for batch in trange(batch_size):
        with torch.no_grad():
            data = skill.execute(env, model, T_exec_max=env.episode_len, plot=True, use_learned_model=True)

if __name__ == "__main__":
    main()