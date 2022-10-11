import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import DataLoader

from Dynamic_GNN_structured_models.datasets import *
from Dynamic_GNN_structured_models.learning import *

def main():

    model_path = './outputs/2022-09-03/11-18-14/EPD_temporal_forward_model_analytical_2objects_modes/'

    checkpoint = torch.load(model_path + 'checkpoints/last.ckpt', map_location='cuda:0')
    cfg_dict = OmegaConf.load(model_path + 'logs/default/version_0/hparams.yaml')

    model_name = cfg_dict['model_name']
    dataset_name = cfg_dict['dataset_name']

    dataset = eval(dataset_name)()

    model = eval(model_name)(
        train_ds_gen=None, 
        val_ds_gen=None, 
        cfg_dict=cfg_dict
        )

    model.load_state_dict(checkpoint['state_dict'])

    batch_size = 1#*64
    n_nodes = 3
    i=2000
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
    
    out_xy_tp1, gt_xy_tp1, gt_xyt, params_list = [], [], [], []

    diff = np.linalg.norm(out-batch.y)
    print(f"Prediction error = {diff}")
    
    for l in range(n_nodes):
        out_xy_tp1.append(out[np.arange(l,out.shape[0], n_nodes),:])
        gt_xy_tp1.append(batch.y[np.arange(l,out.shape[0], n_nodes),:])
        gt_xyt.append(batch.pos[np.arange(l,out.shape[0], n_nodes),:])
        params_list.append(params[np.arange(l,out.shape[0], n_nodes),:])
    control = batch.control[np.arange(0,batch.control.shape[0], n_nodes),:]

    fig, axes = plt.subplots(2, 3, figsize=(18, 15))
    n=4
    indx = 300

    for i in range(n_nodes):
        axes[0, 0].plot(gt_xy_tp1[i][:,0], gt_xy_tp1[i][:,1], label=f'GT object{i}', linewidth=3)
        axes[0, 0].plot(out_xy_tp1[i][:,0], out_xy_tp1[i][:,1], label=f'Pred object{i}', linewidth=1)

        axes[0, 1].plot(gt_xy_tp1[i][:,0], label=f'GT x object{i}')
        axes[0, 1].plot(out_xy_tp1[i][:,0], label=f'Pred x object{i}')

        axes[0, 2].plot(gt_xy_tp1[i][:,1], label=f'GT y object{i}')
        axes[0, 2].plot(out_xy_tp1[i][:,1], label=f'Pred y object{i}')

        axes[1, 0].plot(gt_xy_tp1[i][:,2], label=f'GT x-velocity object{i}')
        axes[1, 0].plot(out_xy_tp1[i][:,2], label=f'Pred x-velocity object{i}')

        axes[1, 1].plot(gt_xy_tp1[i][:,3], label=f'GT y-velocity object{i}')
        axes[1, 1].plot(out_xy_tp1[i][:,3], label=f'Pred y-velocity object{i}')

    axes[0, 0].set_ylabel('y', fontsize=20)
    axes[0, 0].set_xlabel('x', fontsize=20)
    axes[0, 0].set_title('Phase plot', fontsize=20)
    axes[0, 0].legend(prop={'size': 15})

    axes[0, 1].set_ylabel('x', fontsize=20)
    axes[0, 1].set_xlabel('t', fontsize=20)
    axes[0, 1].set_title('x', fontsize=20)
    axes[0, 1].legend(prop={'size': 15})

    axes[0, 2].set_ylabel('y', fontsize=20)
    axes[0, 2].set_xlabel('t', fontsize=20)
    axes[0, 2].set_title('y', fontsize=20)
    axes[0, 2].legend(prop={'size': 15})

    axes[1, 0].set_ylabel('xdot', fontsize=20)
    axes[1, 0].set_xlabel('t', fontsize=20)
    axes[1, 0].set_title('x velocity', fontsize=20)
    axes[1, 0].legend(prop={'size': 15})

    axes[1, 1].set_ylabel('ydot', fontsize=20)
    axes[1, 1].set_xlabel('t', fontsize=20)
    axes[1, 1].set_title('y velocity', fontsize=20)
    axes[1, 1].legend(prop={'size': 15})

    axes[1, 2].plot(control[:,0], label='gripper x control')
    axes[1, 2].plot(control[:,1], label='gripper y control')
    axes[1, 2].set_ylabel('u', fontsize=20)
    axes[1, 2].set_xlabel('t', fontsize=20)
    axes[1, 2].set_title('Control', fontsize=20)
    axes[1, 2].legend(prop={'size': 15})

    plt.show()
    plt.savefig('../Dynamic_GNN_structured_models/tests/media/EPDpredicitonAnalytical3objectsLinearObsMixed.png')
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 8))

    for i in range(n_nodes):

        axes[0, 0].plot(params_list[i][:,0], label=f'1/mx object{i}')
        axes[0, 1].plot(params_list[i][:,1], label=f'1/my object{i}')

        axes[0, 2].plot(params_list[i][:,2], label=f'kx object{i}')
        axes[1, 0].plot(params_list[i][:,3], label=f'ky object{i}')

        axes[1, 1].plot(params_list[i][:,4], label=f'cx object{i}')
        axes[1, 2].plot(params_list[i][:,5], label=f'cy object{i}')

        axes[2, 0].plot(params_list[i][:,6], label=f'x0 object{i}')
        axes[2, 1].plot(params_list[i][:,7], label=f'y0 object{i}')
        
        if cfg_dict["learn_impact_params"]:
            axes[2, 2].plot(params_list[i][:,8], label=f'ix object{i}')
            axes[2, 3].plot(params_list[i][:,9], label=f'iy object{i}')

            # axes[0, 3].plot(params_list[i][:,10], label=f'ixo object{i}')
            # axes[1, 3].plot(params_list[i][:,11], label=f'iyo object{i}')

    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[0, 2].legend()
    axes[0, 3].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 2].legend()
    axes[1, 3].legend()
    axes[2, 0].legend()
    axes[2, 1].legend()
    axes[2, 2].legend()
    axes[2, 3].legend()
    plt.show()
    plt.savefig('../Dynamic_GNN_structured_models/tests/media/params_spring_mass_damper_impact_model.png')
    
    import ipdb; ipdb.set_trace()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    node = 0
    # axes[0,0].plot(gt_xyt[0][:,0], label='GT position-x gripper')
    # axes[0,0].plot(gt_xyt[1][:,0], label='GT position-x object')
    axes[0,0].legend()
    axes[0,0].set_xlabel('t')
    axes[0,0].set_ylabel('Position')
    axes[0,0].set_xlim([0, batch_size])

    # axes[0,1].plot(gt_xyt[0][:,1], label='GT position-y gripper')
    # axes[0,1].plot(gt_xyt[1][:,1], label='GT position-y object')
    axes[0,1].legend()
    axes[0,1].set_xlabel('t')
    axes[0,1].set_ylabel('Position')
    axes[0,1].set_xlim([0, batch_size])

    # axes[0,2].plot(gt_xyt[0][:,2], label='GT velocity-x gripper')
    # axes[0,2].plot(gt_xyt[1][:,2], label='GT velocity-x object')
    axes[0,2].legend()
    axes[0,2].set_xlabel('t')
    axes[0,2].set_ylabel('Position')
    axes[0,2].set_xlim([0, batch_size])

    # axes[0,3].plot(gt_xyt[0][:,3], label='GT velocity-y gripper')
    # axes[0,3].plot(gt_xyt[1][:,3], label='GT velocity-y object')
    axes[0,3].legend()
    axes[0,3].set_xlabel('t')
    axes[0,3].set_ylabel('Position')
    axes[0,3].set_xlim([0, batch_size])

    axes[1,0].plot(latent_xyt[0][:,0], label='latent 0 gripper')
    # axes[1,0].plot(latent_xyt[1][:,0], label='latent 0 object')
    axes[1,0].legend()
    axes[1,0].set_xlabel('t')
    axes[1,0].set_ylabel('latent 0')
    axes[1,0].set_xlim([0, batch_size])

    axes[1,1].plot(latent_xyt[0][:,1], label='latent 1 gripper')
    # axes[1,1].plot(latent_xyt[1][:,1], label='latent 1 object')
    axes[1,1].legend()
    axes[1,1].set_xlabel('t')
    axes[1,1].set_ylabel('latent 1')
    axes[1,1].set_xlim([0, batch_size])

    axes[1,2].plot(latent_xyt[0][:,2], label='latent 2 gripper')
    # axes[1,2].plot(latent_xyt[1][:,2], label='latent 2 object')
    axes[1,2].legend()
    axes[1,2].set_xlabel('t')
    axes[1,2].set_ylabel('latent 2')
    axes[1,2].set_xlim([0, batch_size])

    axes[1,3].plot(latent_xyt[0][:,3], label='latent 3 gripper')
    # axes[1,3].plot(latent_xyt[1][:,3], label='latent 3 object')
    axes[1,3].legend()
    axes[1,3].set_xlabel('t')
    axes[1,3].set_ylabel('latent 3')
    axes[1,3].set_xlim([0, batch_size])

    plt.savefig('/home/saumyas/Projects/Dynamic_GNN_structured_models/tests/media/Visualize_latent_Analytical1obj_E2C.png')
    import ipdb; ipdb.set_trace()
    # --------------------------Only needed if n_envs>1 for ball bounce env-----------------------------------------------------
    n_envs = 64
    out_env = [out_xy[node][np.arange(0, batch_size, n_envs),:] for node in range(n_nodes)]
    gt_env = [gt_xy[node][np.arange(0, batch_size, n_envs),:] for node in range(n_nodes)]

    fig, axes = plt.subplots(2, figsize=(8, 8))
    axes[0].plot(gt_env[0][:,0], label='GT position-y gripper')
    axes[0].plot(out_env[0][:,0], label='pred position-y gripper')
    axes[0].plot(gt_env[1][:,0], label='GT position-y wall')
    axes[0].plot(out_env[1][:,0], label='pred position-y wall')
    axes[0].legend()
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('Position')
    # axes[0].set_xlim([0, batch_size])
    # axes[0].set_ylim([-1, 1])
    
    axes[1].plot(gt_env[0][:,8], label='GT velocity-y gripper')
    axes[1].plot(out_env[0][:,8], label='pred velocity-y gripper')
    axes[1].plot(gt_env[1][:,8], label='GT velocity-y wall')
    axes[1].plot(out_env[1][:,8], label='pred velocity-y wall')
    axes[1].legend()
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('Velocity')
    # axes[1].set_xlim([0, batch_size])
    # axes[1].set_ylim([-1, 1])

    plt.savefig('media/EPDpredicitonBallBounce_env.png')
    
if __name__ == "__main__":
    main()