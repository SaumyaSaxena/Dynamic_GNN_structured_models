import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from Dynamic_GNN_structured_models.controllers.utils import *
import imageio
from isaacgym_utils.math_utils import min_jerk
from torch_geometric.data import Data, Batch, DataLoader
from Dynamic_GNN_structured_models.datasets.data_utils import *

class Skill():
    def __init__(self):
        self.rollout_deltat = 10
        self.t_plan = 10 # Ensure t_plan > rollout_deltat
        self.m=2
        self.return_mode = True

    def generate_parameters(self, env):
        if env.goal_randomize:
            goal_angle = np.random.uniform(low=env.goal_pos_ranges['low'], high=env.goal_pos_ranges['high'])*np.pi/180
        else:
            goal_angle = env.goal_angle*np.pi/180
        
        goal = np.array([env._hinges[0].position[0] + env._height_door/2*np.sin(goal_angle),
                         env._hinges[0].position[1] - env._height_door/2*np.cos(goal_angle)])
        env.goal_pos = goal.copy()
        return goal

    def apply_action(self, env, action):
        return env.step(action)[0]

    def execute(self, env, model, T_exec_max=200, plot=False, use_learned_model=False):
        self.N_O = env.N_O
        x0 = env.reset()
        xf = self.generate_parameters(env)

        self.plan(model, env, x0, xf, 0, T_exec_max, plot=plot, use_learned_model=use_learned_model)
        
        tau_rollout = np.zeros((T_exec_max, env.n+model.m))
        xt = x0.copy()
        images = []
        for t in range(T_exec_max):
            if plot:
                img = env.render(mode='rgb_array')
                images.append(img)
            action = self.get_action(env, xt, t)
            tau_rollout[t] = np.append(xt, action)
            xt = self.apply_action(env, action)
            
        if plot:
            self.plot_obs(tau_rollout, env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout', use_learned_model=use_learned_model)
            # self.plot(self.tau_pred, env.name, ctrl_name=self._ctrl_name+'_tau_pred', plan_latent=use_learned_model)

            # diff = np.linalg.norm(tau_rollout[:,:8]-self.tau_pred[:,:8,0])
            model_name = 'plan_with_latent_model' if use_learned_model else 'plan_with_GT_model'
            imageio.mimsave(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{self._ctrl_name}_u_rollout_{env.name}.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], fps=29)
        return tau_rollout
    
    def execute_mpc(self, env, model, T_exec_max=400, plot=False, use_learned_model=False):
        self.N_O = env.N_O
        x0 = env.reset()
        xf = self.generate_parameters(env)
        
        tau_mpc = np.zeros((T_exec_max, env.n+model.m))
        tau_plan = np.zeros((T_exec_max, model.n*env.N_O+env.m,1))

        n_e = find_edge_index_pickup(self.N_O).shape[1]
        grasps = np.zeros((T_exec_max, n_e//2, 2))
        grasps_pred = np.zeros((T_exec_max, n_e, 2))

        xt = x0.copy()
        images = []
        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            # self.plan(model, env, xt, xf, t*self.rollout_deltat, T_exec_max-t*self.rollout_deltat, plot=False, use_learned_model=use_learned_model)
            self.plan(model, env, xt, xf, t*self.rollout_deltat, self.t_plan, plot=False, use_learned_model=use_learned_model)
            
            if self.return_mode:
                grasps_pred[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.mode_pred[:self.rollout_deltat].copy()
            tau_plan[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.tau_pred[:self.rollout_deltat].copy()
            # tau_latent_plan[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.tau_pred_latent[:self.rollout_deltat].copy()
            for t_roll in range(self.rollout_deltat):
                if plot:
                    img = env.render(mode='rgb_array')
                    images.append(img)
                action = self.get_action(env, xt, t_roll)
                tau_mpc[t*self.rollout_deltat+t_roll] = np.append(xt, action)
                xt = self.apply_action(env, action)

                for b in range(env._num_doors):
                    if env._grasps[b] == 0:
                        grasps[t*self.rollout_deltat+t_roll, b,  1] = 1.
                    else:
                        grasps[t*self.rollout_deltat+t_roll, b,  0] = 1.

        if self.return_mode:
            mode_diff1 = np.linalg.norm(grasps.flatten()-grasps_pred[:,::2,:].flatten())
            mode_diff2 = np.linalg.norm(grasps.flatten()-grasps_pred[:,1::2,:].flatten())
            mode_diff = (mode_diff1+mode_diff2)/2
            print(f'Mode prediction error={mode_diff}')

        diff = np.linalg.norm(tau_mpc[:,:8]-tau_plan[:,:8,0])
        per = diff/np.linalg.norm(tau_mpc[:,:8])*100
        print(f'prediction error={diff}')
        print(f'percentage error={per}')

        if plot:
            self.plot_obs(tau_mpc, env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout_MPC', use_learned_model=use_learned_model)
            self.plot(tau_plan, env.name, ctrl_name=self._ctrl_name+'_tau_pred_MPC', plan_latent=use_learned_model)

            model_name = 'plan_with_latent_model' if use_learned_model else 'plan_with_GT_model'
            imageio.mimsave(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{self._ctrl_name}_u_rollout_MPC_{env.name}.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], fps=29)
        return tau_mpc

    def plot_obs(self, tau, env_name, goal, ctrl_name='iLQR_u_rollout', use_learned_model=False, dof=2):
        tau_gripper = tau[:, :4]
        tau_blocks = tau[:, 4:]
        control = tau[:, -self.m:]
        
        n_blocks = (tau.shape[1] - 4) // 6
        fig, axes = plt.subplots(3, dof, figsize=(25, 15))
        for i in range(dof):
            axes[0, i].plot(tau_gripper[:,i], label=f'gripper pos{i}')
            axes[0, i].plot([goal[i]]*tau.shape[0], label=f'goal pos{i}')
            axes[1, i].plot(tau_gripper[:,dof+i], label=f'gripper velocity{i}')
            
            for j in range(n_blocks):
                axes[0, i].plot(tau_blocks[:,6*j+i], label=f'pos{i} object{j}')
                axes[1, i].plot(tau_blocks[:,6*j+dof+i], label=f'velocity{i} object{j}')
                
            axes[0, i].legend(prop={'size': 15})
            axes[0, i].set_xlabel('t', fontsize=20)

            axes[1, i].legend(prop={'size': 15})
            axes[1, i].set_xlabel('t', fontsize=20)

            axes[2, i].plot(control[:,i], label=f'control{i}')
            axes[2, i].set_xlabel('t')
            axes[2, i].legend(prop={'size': 15})
                
            axes[0, i].set_ylabel('Gripper Position', fontsize=20)
            axes[1, i].set_ylabel('Gripper Velocity', fontsize=20)
            axes[2, i].set_ylabel('Force', fontsize=20)
        
        model_name = 'plan_with_learned_model' if use_learned_model else 'plan_with_GT_model'
        plt.savefig(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{ctrl_name}_{env_name}.png')

    def plot(self, tau_rollout, env_name, ctrl_name='iLQR_u_rollout', plan_latent=False):

        fig, axes = plt.subplots(2, 3, figsize=(15, 15))
        for i in range(self.N_O):
            axes[0, 0].plot(tau_rollout[:,i*4,0], tau_rollout[:,i*4+1,0], label=f'object{i}')
            axes[0, 1].plot(tau_rollout[:,i*4,0], label=f'x object{i}')
            axes[0, 2].plot(tau_rollout[:,i*4+1,0], label=f'y object{i}')
            axes[1, 0].plot(tau_rollout[:,i*4+2,0], label=f'x-velocity object{i}')
            axes[1, 1].plot(tau_rollout[:,i*4+3,0], label=f'y-velocity object{i}')

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

        axes[1, 2].plot(tau_rollout[:,self.N_O*4,0], label='gripper x control')
        axes[1, 2].plot(tau_rollout[:,self.N_O*4+1,0], label='gripper y control')
        axes[1, 2].set_ylabel('u')
        axes[1, 2].set_xlabel('t')
        axes[1, 2].set_title('Control')
        axes[1, 2].legend()

        model_name = 'plan_with_latent_model' if plan_latent else 'plan_with_GT_model'
        plt.savefig(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{ctrl_name}_{env_name}.png')
    
class DoorOpeningPID(Skill):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'Door_opening_PID'
        self.m = 2

        Kp_0 = 5
        self._Ks_0 = np.diag([Kp_0] * 2)
        self._Ds_0 = np.diag([1.0 * np.sqrt(Kp_0)] * 2)

        Kp_1 = 5
        self._Ks_1 = np.diag([Kp_1] * 2)
        self._Ds_1 = np.diag([4 * np.sqrt(Kp_1)] * 2)

    def generate_parameters(self, env):
        if env.goal_randomize:
            if (env._doors[0].position[0] > env._gripper.position[0]):
                goal_angle = np.random.uniform(low=0., high=env.goal_pos_ranges['high'])*np.pi/180
            else:
                goal_angle = np.random.uniform(low=env.goal_pos_ranges['low'], high=0.)*np.pi/180
        else:
            goal_angle = env.goal_angle*np.pi/180
        
        goal = np.array([env._hinges[0].position[0] + env._height_door/2*np.sin(goal_angle),
                         env._hinges[0].position[1] - env._height_door/2*np.cos(goal_angle)])
        env.goal_pos = goal.copy()
        return goal

    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        self._T = T
        _traj1 = [min_jerk(x0[:2], x0[4:6], t, T//2) for t in range(T//2)]
        # _traj1 = [x0[4:6] for t in range(T//2)]
        _traj2 = [min_jerk(x0[4:6], xf[:2], t, T//2) for t in range(T//2)]

        self._traj = _traj1 + _traj2
        self._xf1 = x0[4:6]
        self._xf2 = xf[:2]
    
    def get_action(self, env, xt, t):
        if not env._grasps[0]:
            xe = xt[:2] - self._xf1
            F = -self._Ks_0 @ xe - self._Ds_0 @ xt[2:4]
        else:
            xe = xt[:2] - self._xf2
            F = -self._Ks_1 @ xe - self._Ds_1 @ xt[2:4]
        return F

class DoorOpeningiLQROpenLoop(Skill):
    def __init__(self):
        super().__init__()
        self._thresh_limit = 1e-6
        self._n_ilqr_iter = 20
        self._ctrl_name = 'iLQR_openloop'

    def generate_parameters(self, env):
        if env.goal_randomize:
            if (env._doors[0].position[0] > env._gripper.position[0]):
                goal_angle = np.random.uniform(low=0., high=env.goal_pos_ranges['high'])*np.pi/180
            else:
                goal_angle = np.random.uniform(low=env.goal_pos_ranges['low'], high=0.)*np.pi/180
        else:
            goal_angle = env.goal_angle*np.pi/180
        
        goal_gripper = np.array([env._hinges[0].position[0] + env._height_door/2*np.sin(goal_angle),
                                env._hinges[0].position[1] - env._height_door/2*np.cos(goal_angle)])
        env.goal_pos = goal_gripper.copy()
        
        goal_door = np.concatenate((goal_gripper,
                                    np.zeros(2),
                                    np.array([env._hinges[0].position[0], env._hinges[0].position[1]])
                            ))
        goal_gripper = np.append(goal_gripper, np.zeros(2))
        goal = np.append(goal_gripper, np.tile(goal_door, env._num_doors))
        return goal

    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        x0_cart = env.extract_cartesian_state(x0)
        xf_cart = env.extract_cartesian_state(xf)
        tau_init = env.initial_traj_interpolate(x0_cart, xf_cart, T)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0, xf)

        if t==0:
            env.intialize_cost_matrices(zf, model.n, 0)
        F, f = self.linearize_dynamics_obs_space(model, env, tau_init)
        
        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters, self.mode_pred = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env, return_mode=self.return_mode)
        
        self.tau_pred = model.post_process_tau_latent_to_obs(tau, model.n, is_numpy=False)
        self.control = control.copy()

        if self.return_mode:
            self.mode_pred = self.mode_pred.cpu().numpy()

    def linearize_dynamics_obs_space(self, model, env, tau):
        T = tau.shape[0]
        m=2
        data = []
        for i in range(T):
            x = env.extract_full_state(tau[i, :-m, 0])
            datat = model.ds.data_from_input(x, tau[i, -m:, 0])
            data.append(datat)
        data = Batch.from_data_list(data).to(model._device)
        _, A, B, offset, _ = model._predict_next_state_via_linear_model(data)
        F, f = model.post_process_ABC(A, B, offset)
        return F, f

    def get_action(self, env, xt, t):
        return self.tau_pred[t,4*self.N_O:,:]

class DoorOpeningiLQROpenLoopReactive(Skill):
    def __init__(self):
        super().__init__()
        self._thresh_limit = 1e-6
        self._n_ilqr_iter = 20
        self._ctrl_name = 'iLQR_reactive'
        self.reactive = False
        self.return_mode = True

    def generate_parameters(self, env):
        if env.goal_randomize:
            if (env._doors[0].position[0] > env._gripper.position[0]):
                goal_angle = np.random.uniform(low=0., high=env.goal_pos_ranges['high'])*np.pi/180
            else:
                goal_angle = np.random.uniform(low=env.goal_pos_ranges['low'], high=0.)*np.pi/180
        else:
            goal_angle = env.goal_angle*np.pi/180
        
        goal_gripper = np.array([env._hinges[0].position[0] + env._height_door/2*np.sin(goal_angle),
                                env._hinges[0].position[1] - env._height_door/2*np.cos(goal_angle)])
        env.goal_pos = goal_gripper.copy()
        
        goal_door = np.concatenate((goal_gripper,
                                    np.zeros(2),
                                    np.array([env._hinges[0].position[0], env._hinges[0].position[1]])
                            ))
        goal_gripper = np.append(goal_gripper, np.zeros(2))
        goal = np.append(goal_gripper, np.tile(goal_door, env._num_doors))
        return goal

    def plan(self, model, env, x0, h0, xf, t, T, plot=False, use_learned_model=False):
        x0_cart = env.extract_cartesian_state(x0)
        xf_cart = env.extract_cartesian_state(xf)
        tau_init = env.initial_traj_interpolate(x0_cart, xf_cart, T)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0, xf)

        if t==0:
            env.intialize_cost_matrices(zf, model.n, 0)
        F, f = self.linearize_dynamics_obs_space(model, env, tau_init, h0)

        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters, self.mode_pred = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env, h0=h0, return_mode=self.return_mode)

        self.tau_pred = model.post_process_tau_latent_to_obs(tau, model.n, is_numpy=False)
        self.control = control.copy()
        self.mode_pred = self.mode_pred.cpu().numpy()

    def get_action(self, env, xt, t):
        return self.tau_pred[t,4*self.N_O:,:].reshape(self.m,1)

    def execute_mpc(self, env, model, T_exec_max=400, plot=False, use_learned_model=False):
        
        x0 = env.reset()
        self.N_O = env.N_O
        self.m = env.m
        xf = self.generate_parameters(env)
        
        tau_mpc = np.zeros((T_exec_max, env.n+env.m))
        tau_plan = np.zeros((T_exec_max, model.n*env.N_O+env.m, 1))

        xt = x0.copy()
        images = []
        e_i = find_edge_index_pickup(self.N_O)
        ht = torch.zeros(1,e_i.shape[1], model._cfg_dict["num_edge_types"]).to(model._device)

        n_e = find_edge_index_pickup(self.N_O).shape[1]
        grasps = np.zeros((T_exec_max, n_e//2, 2))
        grasps_pred = np.zeros((T_exec_max, n_e, 2))

        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            
            # self.plan(model, env, xt, ht, xf, t*self.rollout_deltat, T_exec_max-t*self.rollout_deltat, plot=False, use_learned_model=use_learned_model)
            self.plan(model, env, xt, ht, xf, t*self.rollout_deltat, self.t_plan, plot=False, use_learned_model=use_learned_model)

            if self.return_mode:
                grasps_pred[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.mode_pred[:self.rollout_deltat].copy()

            tau_plan[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = self.tau_pred[:self.rollout_deltat].copy()
            for t_roll in range(self.rollout_deltat):
                if plot:
                    img = env.render(mode='rgb_array')
                    images.append(img)
                action = self.get_action(env, xt, t_roll)
                tau_mpc[t*self.rollout_deltat+t_roll] = np.append(xt, action)
                xt = self.apply_action(env, action)
                if self.reactive:
                    ht = model.update_ht_obs_contact(ht, model.process_input_and_final_state(xt, xt)[0], env._grasps[0])
                else:
                    ht = model.update_ht_fwd_prop(ht, model.process_input_and_final_state(xt, xt)[0])
                for b in range(env._num_doors):
                    if env._grasps[b] == 0:
                        grasps[t*self.rollout_deltat+t_roll, b,  1] = 1.
                    else:
                        grasps[t*self.rollout_deltat+t_roll, b,  0] = 1.
        
        if self.return_mode:
            mode_diff1 = np.linalg.norm(grasps.flatten()-grasps_pred[:,::2,:].flatten())
            mode_diff2 = np.linalg.norm(grasps.flatten()-grasps_pred[:,1::2,:].flatten())
            mode_diff = (mode_diff1+mode_diff2)/2
            print(f'Mode prediction error={mode_diff}')


        diff = np.linalg.norm(tau_mpc[:,:8]-tau_plan[:,:8,0])
        per = diff/np.linalg.norm(tau_mpc[:,:8])*100
        print(f'prediction error={diff}')
        print(f'percentage error={per}')

        if plot:
            self.plot_obs(tau_mpc, env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout_MPC', use_learned_model=use_learned_model)
            self.plot(tau_plan, env.name, ctrl_name=self._ctrl_name+'_tau_pred_MPC', plan_latent=use_learned_model)

            
            model_name = 'plan_with_latent_model' if use_learned_model else 'plan_with_GT_model'
            imageio.mimsave(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{self._ctrl_name}_u_rollout_MPC_{env.name}.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], fps=29)
        return tau_mpc
    
    def linearize_dynamics_obs_space(self, model, env, tau, h0):
        T = tau.shape[0]
        m=2
        data = []
        for i in range(T):
            x = env.extract_full_state(tau[i, :-m, 0])
            data.append(np.append(x, tau[i, -m:, 0]))
        tau = np.stack(data, axis=0)

        data_traj = model.ds.data_from_input_traj(tau)
        data = Batch.from_data_list([data_traj]).to(model._device)
        
        _, A, B, offset, _ = model._predict_next_state_via_linear_model(data,ht=h0,T=T)
        F, f = model.post_process_ABC(A, B, offset)
        return F, f
    
    def get_action(self, env, xt, t):
        return self.tau_pred[t,4*self.N_O:,:]