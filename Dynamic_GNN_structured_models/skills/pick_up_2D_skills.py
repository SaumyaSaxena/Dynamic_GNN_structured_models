import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from Dynamic_GNN_structured_models.controllers.utils import *
import imageio
import torch
from Dynamic_GNN_structured_models.datasets.data_utils import *

class Skill():
    def __init__(self):
        self.rollout_deltat = 10
        self.t_plan = 10 # Ensure t_plan > rollout_deltat
        self.return_mode = False

    def generate_parameters(self, env):
        if env.goal_randomize:
            goal_pos = np.random.uniform(low=env.goal_pos_ranges['low'], high=env.goal_pos_ranges['high'])
            goal_vel = np.random.uniform(low=env.goal_vel_ranges['low'], high=env.goal_vel_ranges['high'])
            goal = np.tile(np.append(goal_pos, goal_vel), env.N_O)
            env.goal_pos = goal_pos.copy()
        else:
            goal = np.tile(env.goal_pos + env.goal_vel, env.N_O)
        return goal
    
    def apply_action(self, env, action):
        return env.step(action)[0]
    
    def execute(self, env, model, T_exec_max=400, plot=False, use_learned_model=False):
        x0 = env.reset()
        self.n = env.n
        self.n_latent = model.n
        self.N_O = env.N_O
        self.m = env.m
        xf = self.generate_parameters(env)
        self.plan(model, env, x0[:,None], xf, 0, T_exec_max, plot=plot, use_learned_model=use_learned_model)
        tau_rollout = np.zeros((T_exec_max, env.n*env.N_O+env.m, 1))
        xt = x0[:,None]
        images = []
        for t in range(T_exec_max):
            if plot and env.name == 'BlocksGraspXZ':
                img = env.render(mode='rgb_array')
                images.append(img)
            action = self.get_action(env, xt, t)
            tau_rollout[t] = np.append(xt, action, axis = 0)
            xt = self.apply_action(env, action)
            
        if plot:
            self.plot(tau_rollout, env.name, ctrl_name=self._ctrl_name+'_u_rollout', plan_latent=use_learned_model)
            self.plot(self.tau_pred, env.name, ctrl_name=self._ctrl_name+'_tau_pred', plan_latent=use_learned_model)
            
            if env.name == 'BlocksGraspXZ':
                model_name = 'plan_with_latent_model' if use_learned_model else 'plan_with_GT_model'
                imageio.mimsave(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{self._ctrl_name}_u_rollout_{env.name}.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], fps=29)
        return tau_rollout
    
    def execute_mpc(self, env, model, T_exec_max=400, plot=False, use_learned_model=False):
        x0 = env.reset()
        self.n = env.n
        self.n_latent = model.n
        self.N_O = env.N_O
        self.m = env.m
        xf = self.generate_parameters(env)
        
        tau_mpc = np.zeros((T_exec_max, env.n*env.N_O+env.m, 1))
        tau_plan = np.zeros((T_exec_max, env.n*env.N_O+env.m, 1))

        n_e = find_edge_index_pickup(self.N_O).shape[1]
        grasps = np.zeros((T_exec_max, n_e//2, 2))
        grasps_pred = np.zeros((T_exec_max, n_e, 2))

        xt = x0[:,None]
        images = []
        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            # self.plan(model, env, xt, xf, t*self.rollout_deltat, T_exec_max-t*self.rollout_deltat, plot=False, use_learned_model=use_learned_model)
            self.plan(model, env, xt, xf, t*self.rollout_deltat, self.t_plan, plot=False, use_learned_model=use_learned_model)
            if self.return_mode:
                grasps_pred[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.mode_pred[:self.rollout_deltat].copy()
            tau_plan[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.tau_pred[:self.rollout_deltat].copy()
            
            for t_roll in range(self.rollout_deltat):
                if plot and env.name == 'BlocksGraspXZ':
                    img = env.render(mode='rgb_array')
                    images.append(img)
                action = self.get_action(env, xt, t_roll)
                tau_mpc[t*self.rollout_deltat+t_roll] = np.append(xt, action, axis = 0)

                xt = self.apply_action(env, action)
                for b in range(env._num_blocks):
                    if env._grasps[b] == 0:
                        grasps[t*self.rollout_deltat+t_roll, b,  1] = 1.
                    else:
                        grasps[t*self.rollout_deltat+t_roll, b,  0] = 1.

        if self.return_mode:
            mode_diff1 = np.linalg.norm(grasps.flatten()-grasps_pred[:,::2,:].flatten())
            mode_diff2 = np.linalg.norm(grasps.flatten()-grasps_pred[:,1::2,:].flatten())
            mode_diff = (mode_diff1+mode_diff2)/2
            print(f'Mode prediction error={mode_diff}')

        diff = np.linalg.norm(tau_mpc[:,:2,0]-tau_plan[:,:2,0])
        per = diff/np.linalg.norm(tau_mpc[:,:2,0])*100
        print(f'prediction error={diff}')
        print(f'percentage error={per}')
        if plot:
            self.plot(tau_mpc, env.name, ctrl_name=self._ctrl_name+'_u_rollout_MPC', plan_latent=use_learned_model)
            self.plot(tau_plan, env.name, ctrl_name=self._ctrl_name+'_tau_pred_MPC', plan_latent=use_learned_model)
            
            if env.name == 'BlocksGraspXZ':
                model_name = 'plan_with_latent_model' if use_learned_model else 'plan_with_GT_model'
                imageio.mimsave(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{self._ctrl_name}_u_rollout_MPC_{env.name}.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], fps=29)
        return tau_mpc

    def plot(self, tau_rollout, env_name, ctrl_name='iLQR_u_rollout', plan_latent=False):
        if env_name == 'PickUp2DGymEnv':
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes[0, 0].plot(tau_rollout[:,0,0], tau_rollout[:,1,0], label='gripper')
            axes[0, 0].plot(tau_rollout[:,4,0], tau_rollout[:,5,0], label='object')
            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_title('Phase plot')
            axes[0, 0].legend()

            axes[0, 1].plot(tau_rollout[:,0,0], label='x gripper')
            axes[0, 1].plot(tau_rollout[:,4,0], label='x object')
            axes[0, 1].set_ylabel('x')
            axes[0, 1].set_xlabel('t')
            axes[0, 1].set_title('x')
            axes[0, 1].legend()

            axes[0, 2].plot(tau_rollout[:,1,0], label='y gripper')
            axes[0, 2].plot(tau_rollout[:,5,0], label='y object')
            axes[0, 2].set_ylabel('y')
            axes[0, 2].set_xlabel('t')
            axes[0, 2].set_title('y')
            axes[0, 2].legend()

            axes[1, 0].plot(tau_rollout[:,2,0], label='gripper x velocity')
            axes[1, 0].plot(tau_rollout[:,6,0], label='object x velocity')
            axes[1, 0].set_ylabel('xdot')
            axes[1, 0].set_xlabel('t')
            axes[1, 0].set_title('x velocity')
            axes[1, 0].legend()

            axes[1, 1].plot(tau_rollout[:,3,0], label='gripper y velocity')
            axes[1, 1].plot(tau_rollout[:,7,0], label='object y velocity')
            axes[1, 1].set_ylabel('ydot')
            axes[1, 1].set_xlabel('t')
            axes[1, 1].set_title('y velocity')
            axes[1, 1].legend()

            axes[1, 2].plot(tau_rollout[:,8,0], label='gripper x control')
            axes[1, 2].plot(tau_rollout[:,9,0], label='gripper y control')
            axes[1, 2].set_ylabel('u')
            axes[1, 2].set_xlabel('t')
            axes[1, 2].set_title('Control')
            axes[1, 2].legend()
        
        if env_name == 'BlocksGraspXZ':
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            for i in range(self.N_O):
                axes[0, 0].plot(tau_rollout[:,i*self.n,0], tau_rollout[:,i*self.n+1,0], label=f'object{i}')
                axes[0, 1].plot(tau_rollout[:,i*self.n,0], label=f'x object{i}')
                axes[0, 2].plot(tau_rollout[:,i*self.n+1,0], label=f'y object{i}')
                axes[1, 0].plot(tau_rollout[:,i*self.n+2,0], label=f'x-velocity object{i}')
                axes[1, 1].plot(tau_rollout[:,i*self.n+3,0], label=f'y-velocity object{i}')

            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_title('Phase plot')
            axes[0, 0].legend()

            axes[0, 1].set_ylabel('x')
            axes[0, 1].set_xlabel('t')
            axes[0, 1].set_title('x')
            axes[0, 1].legend()

            axes[0, 2].set_ylabel('y')
            axes[0, 2].set_xlabel('t')
            axes[0, 2].set_title('y')
            axes[0, 2].legend()

            axes[1, 0].set_ylabel('xdot')
            axes[1, 0].set_xlabel('t')
            axes[1, 0].set_title('x velocity')
            axes[1, 0].legend()

            axes[1, 1].set_ylabel('ydot')
            axes[1, 1].set_xlabel('t')
            axes[1, 1].set_title('y velocity')
            axes[1, 1].legend()

            axes[1, 2].plot(tau_rollout[:,self.N_O*self.n,0], label='gripper x control')
            axes[1, 2].plot(tau_rollout[:,self.N_O*self.n+1,0], label='gripper y control')
            axes[1, 2].set_ylabel('u')
            axes[1, 2].set_xlabel('t')
            axes[1, 2].set_title('Control')
            axes[1, 2].legend()

        model_name = 'plan_with_latent_model' if plan_latent else 'plan_with_GT_model'
        plt.savefig(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{ctrl_name}_{env_name}.png')
    

class PickToGoaliLQROpenLoop(Skill):
    def __init__(self):
        super().__init__()
        self._thresh_limit = 1e-6
        self._n_ilqr_iter = 20
        self._ctrl_name = 'iLQR_openloop'
        
    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        tau_init = env.initial_traj_interpolate(x0, xf, T)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0, xf)
        xf1 = np.tile(x0[env.n:2*env.n,:], (env.N_O,1))
        datazf1, datazf2, zf1, zf2 = model.process_input_and_final_state(xf1, xf)

        if t==0:
            pick_up_seq = env._cfg['gym']['goal']['indices_blocks_to_pick']
            env.intialize_cost_matrices(zf2, model.n, pick_up_seq)
        F, f = model.linearize_dynamics_obs_space(tau_init)
        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters, self.mode_pred = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env, return_mode=self.return_mode)

        self.tau_pred = model.post_process_tau_latent_to_obs(tau, env.n, is_numpy=False)
        self.control = control.copy()
        if self.return_mode:
            self.mode_pred = self.mode_pred.cpu().numpy()

        if plot:
            self.plot(self.tau_pred, env.name, ctrl_name=self._ctrl_name+'_tau_pred', plan_latent=use_learned_model)

    def get_action(self, env, xt, t):
        return self.tau_pred[t,self.n*self.N_O:,:].reshape(self.m,1)

class PickToGoaliLQRClosedLoop(PickToGoaliLQROpenLoop):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'iLQR_closedloop'
    
    def get_action(self, env, xt, t):
        return self.control['K'][t]@xt + self.control['k'][t]


class PickupPID(Skill):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'Pickup_PID'
        self.m = 2

        Kp_0 = 5
        self._Ks_0 = np.diag([Kp_0] * 2)
        self._Ds_0 = np.diag([0.5 * np.sqrt(Kp_0)] * 2)

        Kp_1 = 5
        self._Ks_1 = np.diag([Kp_1] * 2)
        self._Ds_1 = np.diag([1 * np.sqrt(Kp_1)] * 2)

        Kp_2 = 5
        self._Ks_2 = np.diag([Kp_2] * 2)
        self._Ds_2 = np.diag([4 * np.sqrt(Kp_2)] * 2)

        self._num_objs_pick = 2

    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        self._T = T
        if env._num_blocks < self._num_objs_pick:
            raise ValueError(f"Invalid num objects to pick: {self._num_objs_pick}")
        
        self._xf_obj = []
        for i in range(self._num_objs_pick):
            self._xf_obj.append(np.array([env._blocks[i].position[0], 
                                        env._blocks[i].position[1]]))

        self._xf = xf[:2]

    def get_action(self, env, xt, t):
        if (not env._grasps[0]) and (not env._grasps[1]): # none grasped
            xe = xt[:2,0] - self._xf_obj[0]
            F = -self._Ks_0 @ xe - self._Ds_0 @ xt[2:4,0]

        if (env._grasps[0]) and (not env._grasps[1]): # first object grasped
            xe = xt[:2,0] - self._xf_obj[1]
            F = -self._Ks_1 @ xe - self._Ds_1 @ xt[2:4,0]
        
        if (not env._grasps[0]) and (env._grasps[1]): # second object grasped
            xe = xt[:2,0] - self._xf_obj[0]
            F = -self._Ks_1 @ xe - self._Ds_1 @ xt[2:4,0]

        if env._grasps[0] and env._grasps[1]: # both grasped , goto goal
            xe = xt[:2,0] - self._xf
            F = -self._Ks_2 @ xe - self._Ds_2 @ xt[2:4,0]
        return F.reshape(self.m,1)


class PickToGoaliLQROpenLoopReactive(Skill):
    def __init__(self):
        super().__init__()
        self._thresh_limit = 1e-6
        self._n_ilqr_iter = 20
        self._ctrl_name = 'iLQR_reactive'
        self.reactive = True
        self.return_mode = True

    def plan(self, model, env, x0, h0, xf, t, T, plot=False, use_learned_model=False):
        tau_init = env.initial_traj_interpolate(x0, xf, T)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0, xf)
        xf1 = np.tile(x0[env.n:2*env.n,:], (env.N_O,1))
        datazf1, datazf2, zf1, zf2 = model.process_input_and_final_state(xf1, xf)

        if t==0:
            pick_up_seq = env._cfg['gym']['goal']['indices_blocks_to_pick']
            env.intialize_cost_matrices(zf2, model.n, pick_up_seq)
        F, f = model.linearize_dynamics_obs_space(tau_init,h0,T)

        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters, self.mode_pred = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env, h0=h0, return_mode=self.return_mode)

        self.tau_pred = model.post_process_tau_latent_to_obs(tau, env.n, is_numpy=False)
        self.control = control.copy()

        self.mode_pred = self.mode_pred.cpu().numpy()

        if plot:
            self.plot(self.tau_pred, env.name, ctrl_name=self._ctrl_name+'_tau_pred', plan_latent=use_learned_model)

    def get_action(self, env, xt, t):
        return self.tau_pred[t,self.n*self.N_O:,:].reshape(self.m,1)

    def execute_mpc(self, env, model, T_exec_max=400, plot=False, use_learned_model=False):
        
        x0 = env.reset()
        self.n = env.n
        self.n_latent = model.n
        self.N_O = env.N_O
        self.m = env.m
        xf = self.generate_parameters(env)
        
        tau_mpc = np.zeros((T_exec_max, env.n*env.N_O+env.m, 1))
        tau_plan = np.zeros((T_exec_max, env.n*env.N_O+env.m, 1))
        
        n_e = find_edge_index_pickup(self.N_O).shape[1]
        grasps = np.zeros((T_exec_max, n_e//2, 2))
        grasps_pred = np.zeros((T_exec_max, n_e, 2))

        xt = x0[:,None]
        images = []

        ht = torch.zeros(1, n_e, model._cfg_dict["num_edge_types"]).to(model._device)

        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            # self.plan(model, env, xt, ht, xf, t*self.rollout_deltat, T_exec_max-t*self.rollout_deltat, plot=False, use_learned_model=use_learned_model)
            self.plan(model, env, xt, ht, xf, t*self.rollout_deltat, self.t_plan, plot=False, use_learned_model=use_learned_model)
            if self.return_mode:
                grasps_pred[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.mode_pred[:self.rollout_deltat].copy()
            tau_plan[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.tau_pred[:self.rollout_deltat].copy()
            # tau_latent_plan[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.tau_pred_latent[:self.rollout_deltat].copy()
            for t_roll in range(self.rollout_deltat):
                if plot and env.name == 'BlocksGraspXZ':
                    img = env.render(mode='rgb_array')
                    images.append(img)
                action = self.get_action(env, xt, t_roll)
                tau_mpc[t*self.rollout_deltat+t_roll] = np.append(xt, action, axis = 0)
                xt = self.apply_action(env, action)
                if self.reactive:
                    ht = model.update_ht_obs_contact(ht, model.process_input_and_final_state(xt, xt)[0], env._grasps)
                else:
                    ht = model.update_ht_fwd_prop(ht, model.process_input_and_final_state(xt, xt)[0])
                
                for b in range(env._num_blocks):
                    if env._grasps[b] == 0:
                        grasps[t*self.rollout_deltat+t_roll, b,  1] = 1.
                    else:
                        grasps[t*self.rollout_deltat+t_roll, b,  0] = 1.

        if self.return_mode:
            mode_diff1 = np.linalg.norm(grasps.flatten()-grasps_pred[:,::2,:].flatten())
            mode_diff2 = np.linalg.norm(grasps.flatten()-grasps_pred[:,1::2,:].flatten())
            mode_diff = (mode_diff1+mode_diff2)/2
            print(f'Mode prediction error={mode_diff}')
        
        diff = np.linalg.norm(tau_mpc[:,:,0]-tau_plan[:,:,0])
        per = diff/np.linalg.norm(tau_mpc[:,:,0])*100
        print(f'prediction error={diff}')
        print(f'percentage error={per}')
        if plot:
            self.plot(tau_mpc, env.name, ctrl_name=self._ctrl_name+'_u_rollout_MPC', plan_latent=use_learned_model)
            self.plot(tau_plan, env.name, ctrl_name=self._ctrl_name+'_tau_pred_MPC', plan_latent=use_learned_model)

            if env.name == 'BlocksGraspXZ':
                model_name = 'plan_with_latent_model' if use_learned_model else 'plan_with_GT_model'
                imageio.mimsave(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{self._ctrl_name}_u_rollout_MPC_{env.name}.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], fps=29)
        return tau_mpc