import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import DataLoader

from Dynamic_GNN_structured_models.datasets import *
from Dynamic_GNN_structured_models.learning import *

torch.manual_seed(0)
np.random.seed(0)

def main():

    model_path = './outputs/2022-02-27/05-04-27/EPD_linear_obs_spring_mass_damper_model_isaac_franka_sliding/'

    checkpoint = torch.load(model_path + 'checkpoints/last.ckpt', map_location='cuda:0')
    cfg_dict = OmegaConf.load(model_path + 'logs/default/version_0/hparams.yaml')

    model_name = cfg_dict['model_name']
    dataset_name = cfg_dict['dataset_name']

    dataset = eval(dataset_name)()

    cfg_dict['gpu'] = 0 # to enable testing on machines with 1 gpu
    model = eval(model_name)(
        train_ds_gen=None, 
        val_ds_gen=None, 
        cfg_dict=cfg_dict
        )
    
    model.load_state_dict(checkpoint['state_dict'])
    batch_size = 400
    i=0
    # i=1200
    indices = range(i*batch_size,(i+1)*batch_size)  # select your indices here as a list  
    subset = torch.utils.data.Subset(dataset, indices)
    
    gpu_load = cfg_dict.get('gpu', 0)
    device = f'cuda:{gpu_load}'
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader)).to(device)

    model.eval()
    with torch.no_grad():
        out, params = model(batch)

    out = out.cpu()
    batch = batch.cpu()
    params = params.cpu()

    dof = cfg_dict['dof']
    
    out_tp1, gt_tp1, params_list = [], [], []
    n_nodes = model._N_O
    for l in range(n_nodes):
        out_tp1.append(out[l::n_nodes,:])
        gt_tp1.append(batch.y[l::n_nodes,:])
        params_list.append(params[l::n_nodes,:])
    control = batch.control[0::n_nodes,:]

    fig, axes = plt.subplots(3, dof, figsize=(25, 15))
    for i in range(dof):
        for j in range(n_nodes):
            axes[0, i].plot(gt_tp1[j][:,i], label=f'GT pos{i} object{j}')
            axes[0, i].plot(out_tp1[j][:,i], label=f'Pred pos{i} object{j}')
            axes[0, i].set_xlabel('t', fontsize=20)
            axes[0, i].legend(prop={'size': 15})

            axes[1, i].plot(gt_tp1[j][:,i+dof], label=f'GT velocity{i} object{j}')
            axes[1, i].plot(out_tp1[j][:,i+dof], label=f'Pred velocity{i} object{j}')
            axes[1, i].set_xlabel('t', fontsize=20)
            axes[1, i].legend(prop={'size': 15})

        axes[2, i].plot(control[:,i], label=f'GT control{i}')
        axes[2, i].set_xlabel('t', fontsize=20)
        axes[2, i].legend(prop={'size': 15})

        if dof ==7:
            axes[0, i].set_ylabel('Theta', fontsize=20)
            axes[1, i].set_ylabel('Thetadot', fontsize=20)
            axes[2, i].set_ylabel('Torque', fontsize=20)
        else:
            axes[0, i].set_ylabel('Gripper Position', fontsize=20)
            axes[1, i].set_ylabel('Gripper Velocity', fontsize=20)
            axes[2, i].set_ylabel('Force', fontsize=20)

    plt.savefig('../Dynamic_GNN_structured_models/tests/media/FrankaPickupEPDprediciton.png')
    
    fig, axes = plt.subplots(4, dof, figsize=(25, 15))
    for i in range(dof):
        for j in range(n_nodes):
            axes[0, i].plot(params_list[j][:,i], label=f'1/m dof{i} object{j}')
            axes[1, i].plot(params_list[j][:,dof+i], label=f'k dof{i} object{j}')
            axes[2, i].plot(params_list[j][:,2*dof+i], label=f'c dof{i} object{j}')
            axes[3, i].plot(params_list[j][:,3*dof+i], label=f'x0 dof{i} object{j}')
            
            axes[0, i].legend(prop={'size': 15})
            axes[1, i].legend(prop={'size': 15})
            axes[2, i].legend(prop={'size': 15})
            axes[3, i].legend(prop={'size': 15})
            
            if cfg_dict["learn_impact_params"]:
                axes[4, i].plot(params_list[j][:,4*dof+i], label=f'i dof{i} object{j}')
                axes[4, i].legend(prop={'size': 15})
            
                # axes[5, i].plot(params_list[j][:,5*dof+i], label=f'ixo dof{i} object{j}')
                # axes[5, i].legend()
        
    plt.savefig('../Dynamic_GNN_structured_models/tests/media/franka_params_spring_mass_damper_model.png')

if __name__ == "__main__":
    main()