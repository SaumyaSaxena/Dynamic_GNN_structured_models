import numpy as np
from tqdm import trange
from isaacgym import gymapi
from isaacgym_utils.policy import EEImpedanceWaypointPolicy
from tqdm import trange
from Dynamic_GNN_structured_models.controllers.utils import *
from isaacgym_utils.math_utils import quat_to_np, angle_axis_between_quats, vec3_to_np, np_to_vec3, min_jerk, slerp_quat, \
    compute_task_space_impedance_control
import quaternion
import torch
from Dynamic_GNN_structured_models.datasets.data_utils import *

class Skill():
    def __init__(self, cfg_dict=None):
        self._cfg_dict = cfg_dict
        self.rollout_deltat = 10
        self.t_plan = 10 # Ensure t_plan > rollout_deltat
        self.return_mode = False

    def generate_parameters(self, env):
        if self._cfg_dict['goal']['randomize']:
            goal_pos = np.random.uniform(low=self._cfg_dict['goal']['position_ranges']['low'], high=self._cfg_dict['goal']['position_ranges']['high'])
        else:
            goal_pos = np.array(self._cfg_dict['goal']['position'])
        return goal_pos
    
    def apply_actions(self, env, actions):
        return env.step(actions)[0]
    
    def get_actions(self, env, xt, t):
        pass
    
    def execute(self, env, model, T_exec_max=300, plot=False, use_learned_model=False):
        if use_learned_model:
            self.dof = model._dof

        env.init_joints = self._cfg_dict['initial_joints']
        x0 = env.reset()
        xf = self.generate_parameters(env)

        self.plan(model, env, x0, xf, 0, T_exec_max, plot=plot, use_learned_model=use_learned_model)
        tau_rollout = [np.zeros((T_exec_max, env.n+self.m))]*env._n_envs
        print("Planning done. Executing...")
        xt = x0.copy()
        for t in trange(T_exec_max):
            actions = self.get_actions(env, xt, t)
            for env_idx in range(env._n_envs):
                tau_rollout[env_idx][t] = np.append(xt[env_idx], actions[env_idx])
                if env._num_blocks > 0:
                    dist_to_block = np.linalg.norm(xt[env_idx][14:17] - self.init_block_pos[:3])
                    if dist_to_block < self._thresh_dist:
                        env._franka.close_grippers(env_idx, env._franka_name)
            xt = self.apply_actions(env, actions)
        
        if plot:
            self.plot(tau_rollout[0], env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout', use_learned_model=use_learned_model, dof=self.dof)
            self.plot(self.tau_pred, env.name, xf, ctrl_name=self._ctrl_name+'_tau_pred', use_learned_model=use_learned_model, dof=self.dof)
    
        return tau_rollout
    
    def execute_mpc(self, env, model, T_exec_max=300, plot=False, use_learned_model=False):
        if use_learned_model:
            self.dof = model._cfg_dict['dof']
        env.init_joints = self._cfg_dict['initial_joints']
        x0 = env.reset()
        xf = self.generate_parameters(env)

        tau_mpc = np.zeros((T_exec_max, env.n+self.m))
        tau_plan = []

        xt = x0.copy()
        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            # self.plan(model, env, xt, xf, t*self.rollout_deltat, T_exec_max-t*self.rollout_deltat, plot=False, use_learned_model=use_learned_model)
            self.plan(model, env, xt, xf, t*self.rollout_deltat, self.t_plan, plot=False, use_learned_model=use_learned_model)
            tau_plan.append(self.tau_pred[:self.rollout_deltat])
            for t_roll in range(self.rollout_deltat):
                actions = self.get_actions(env, xt, t_roll)
                tau_mpc[t*self.rollout_deltat+t_roll] = np.append(xt[0], actions[0])
                if env._num_blocks > 0:
                    dist_to_block = np.linalg.norm(xt[0][14:17] - self.init_block_pos[:3])
                    # print(dist_to_block)
                    # all_ct_forces = env._scene.gym.get_rigid_contact_forces(env._scene.sim)
                    # all_ct = env._scene.gym.get_rigid_contacts(env._scene.sim)
                    # env._scene.gym.get_actor_index(env,env._franka,0)
                    # import ipdb; ipdb.set_trace()
                    if dist_to_block < self._thresh_dist:
                        env._franka.close_grippers(0, env._franka_name)
                xt = self.apply_actions(env, actions)
        tau_plan = np.concatenate(tau_plan, axis=0)

        diff = self.find_prediction_error(env, tau_plan, tau_mpc)
        print(f"Prediction error={diff}")
        import ipdb; ipdb.set_trace()
        if plot:
            self.plot(tau_mpc, env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout_MPC', use_learned_model=use_learned_model, dof=self.dof)
            self.plot(tau_plan, env.name, xf, ctrl_name=self._ctrl_name+'_tau_pred_MPC', use_learned_model=use_learned_model, dof=self.dof)

        # Executing again
        x0 = env.reset()
        if env._num_blocks > 0:
            self.init_block_pos = x0[0][27:27+13]
        for t in trange(T_exec_max):
            xt = self.apply_actions(env, [tau_mpc[t,-self.m:]])
            if env._num_blocks > 0:
                dist_to_block = np.linalg.norm(xt[0][14:17] - self.init_block_pos[:3])
                # print(dist_to_block)
                if dist_to_block < self._thresh_dist:
                    env._franka.close_grippers(0, env._franka_name)
        return tau_mpc
    
    def find_prediction_error(self, env, tau_pred, tau):
        diff=0
        for i in range(tau.shape[0]):
            x = env.extract_cartesian_state(tau[i,:-self.m])
            diff += np.linalg.norm(x - tau_pred[i, :-3, 0])
        diff = diff/tau.shape[0]
        return diff

    def plot(self, tau, env_name, goal, ctrl_name='iLQR_u_rollout', use_learned_model=False, dof=7):
        n_blocks = 1
        if tau.shape[1] > dof+(n_blocks+1)*2*dof and dof==3: # rollout trajs
            x = np.concatenate((tau[:, 14:17], tau[:,21:24]), axis=1)
            for i in range(0, n_blocks):
                x = np.concatenate((x, np.concatenate((tau[:, 27+13*i:27+13*i+3], tau[: ,27+13*i+7:27+13*i+10]), axis=1)), axis=1)
            u = tau[:, -6:-3]
        else:
            x = tau[:, :-dof, 0]
            u = tau[:, -dof:, 0]

        fig, axes = plt.subplots(3, dof, figsize=(25, 15))
        for i in range(dof):
            for j in range(n_blocks):
                axes[0, i].plot(x[:,j*2*dof+i], label=f'pos{i} object{j}')
                axes[0, i].plot([goal[i]]*tau.shape[0], label=f'Goal{i}', linewidth=3)

                axes[1, i].plot(x[:,j*2*dof+i+dof], label=f'velocity{i} object{j}')
                
            axes[0, i].legend(prop={'size': 15})
            axes[0, i].set_xlabel('t', fontsize=20)

            axes[1, i].legend(prop={'size': 15})
            axes[1, i].set_xlabel('t', fontsize=20)

            axes[2, i].plot(u[:,i], label=f'control{i}')
            axes[2, i].set_xlabel('t')
            axes[2, i].legend(prop={'size': 15})
                
            if dof ==7:
                axes[0, i].set_ylabel('Theta', fontsize=20)
                axes[1, i].set_ylabel('Thetadot', fontsize=20)
                axes[2, i].set_ylabel('Torque', fontsize=20)
            else:
                axes[0, i].set_ylabel('Gripper Position', fontsize=20)
                axes[1, i].set_ylabel('Gripper Velocity', fontsize=20)
                axes[2, i].set_ylabel('Force', fontsize=20)
        
        model_name = 'plan_with_learned_model' if use_learned_model else 'plan_with_GT_model'
        plt.savefig(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{ctrl_name}_{env_name}.png')
            

class FrankaEEImpedanceControlFreeSpace(Skill):
    def __init__(self, cfg_dict=None):
        super(FrankaEEImpedanceControlFreeSpace, self).__init__(cfg_dict=cfg_dict)
        self.m = 13
    
    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)
        goal_ee_transform = gymapi.Transform(
            p=gymapi.Vec3(xf[0], xf[1], xf[2]),
            r=init_ee_transform.r
        )
        env.goal_ee_transform = goal_ee_transform
        self._policy = EEImpedanceWaypointPolicy(env._franka, env._franka_name, init_ee_transform, goal_ee_transform, T=T) # TODO: don't have same goal for all envs

    def get_actions(self, env, xt, t):
        target_transform = self._policy._traj[min(t, self._policy._T - 1)]
        
        taus = []
        for env_idx in range(env._n_envs):
            tau, F = self._policy._ee_impedance_ctrlr.compute_tau(env_idx, target_transform)
            taus.append(np.append(tau,F))
        taus = np.stack(taus, axis=0)
        return taus
    
    def apply_actions(self, env, actions):
        return env.step(actions[:, :7])[0]


class FrankaEEImpedanceControlPickUp(Skill):
    def __init__(self, cfg_dict=None):
        super(FrankaEEImpedanceControlPickUp, self).__init__(cfg_dict=cfg_dict)
        self.m = 13
        self._ctrl_name = 'Impedance_control_pickup'

    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)
        goal_ee_transform = gymapi.Transform(
            p=gymapi.Vec3(xf[0], xf[1], xf[2]),
            r=init_ee_transform.r
        )
        env.goal_ee_transform = goal_ee_transform
        self._policy = EEImpedanceWaypointPolicy(env._franka, env._franka_name, init_ee_transform, goal_ee_transform, T=T) # TODO: don't have same goal for all envs

    def execute(self, env, model, T_exec_max=300, plot=False, use_learned_model=False):
        env.init_joints = self._cfg_dict['initial_joints']
        x0 = env.reset()
        
        # Goto block
        block_pos = x0[0][27+self._cfg_dict['indx_block_to_pick']*13:27+(self._cfg_dict['indx_block_to_pick']+1)*13]
        block_pos[:3] += [0., -0.01, 0.]
        self.plan(model, env, x0, block_pos[:3], 0, T_exec_max//2, plot=plot, use_learned_model=use_learned_model)
        
        tau_rollout = [np.zeros((T_exec_max, env.n+self.m))]*env._n_envs
        xt = x0.copy()
        for t in trange(T_exec_max//2):
            actions = self.get_actions(env, xt, t)
            for env_idx in range(env._n_envs):
                tau_rollout[env_idx][t] = np.append(xt[env_idx], actions[env_idx])
            xt = self.apply_actions(env, actions)
        
        # Grasp
        for env_idx in range(env._n_envs):
            env._franka.close_grippers(env_idx, env._franka_name)

        # Goto goal
        xf = self.generate_parameters(env)
        self.plan(model, env, x0, xf, T_exec_max//2, T_exec_max-T_exec_max//2, plot=plot, use_learned_model=use_learned_model)
        
        for t in trange(T_exec_max//2, T_exec_max):
            actions = self.get_actions(env, xt, t-T_exec_max//2)
            for env_idx in range(env._n_envs):
                tau_rollout[env_idx][t] = np.append(xt[env_idx], actions[env_idx])
            xt = self.apply_actions(env, actions)
        
        if plot:
            self.plot(tau_rollout[0], env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout', use_learned_model=use_learned_model, dof=3)

        return tau_rollout

    def get_actions(self, env, xt, t):
        target_transform = self._policy._traj[min(t, self._policy._T - 1)]
        
        taus = []
        for env_idx in range(env._n_envs):
            tau, F = self._policy._ee_impedance_ctrlr.compute_tau(env_idx, target_transform)
            taus.append(np.append(tau,F))
        taus = np.stack(taus, axis=0)
        return taus

    def plot(self, tau, env_name, goal, ctrl_name='iLQR_u_rollout', use_learned_model=False, dof=7):
        FrankaEEImpedanceControlDynamicPickUp.plot(FrankaEEImpedanceControlDynamicPickUp, tau, env_name, goal, ctrl_name=ctrl_name, use_learned_model=use_learned_model, dof=dof)

class FrankaEEImpedanceControlDynamicPickUp(Skill):
    def __init__(self, cfg_dict=None):
        super(FrankaEEImpedanceControlDynamicPickUp, self).__init__(cfg_dict=cfg_dict)
        self.m = 13
        self._ctrl_name = 'Impedance_control_dynamic_pickup'

        Kp_0 = [600, 100, 600]
        Kr_0 = [8, 8, 8]
        self._Ks_0 = np.diag(Kp_0 + Kr_0)

        Kp_d_0 = [4 * np.sqrt(Kp_0[0]), 4 * np.sqrt(Kp_0[1]), 4 * np.sqrt(Kp_0[2])]
        Kr_d_0 = [2 * np.sqrt(Kr_0[0]), 2 * np.sqrt(Kr_0[1]), 2 * np.sqrt(Kr_0[2])]
        self._Ds_0 = np.diag(Kp_d_0 + Kr_d_0)

        self._thresh_dist = 0.015
    
    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        self._T = T
        init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)

        self.init_block_pos = x0[0][27+self._cfg_dict['indx_block_to_pick']*13:27+(self._cfg_dict['indx_block_to_pick']+1)*13]

        goal_ee_transform = gymapi.Transform(
            p=gymapi.Vec3(xf[0], xf[1], xf[2]),
            r=init_ee_transform.r
        )
        env.goal_ee_transform = goal_ee_transform

        init_ee_pos = vec3_to_np(init_ee_transform.p)
        block_ee_pos = np.array(self.init_block_pos[:3])
        goal_ee_pos = vec3_to_np(goal_ee_transform.p)
        _traj1 = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(init_ee_pos, block_ee_pos, t, T//2)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, T//2),
            )
            for t in range(T//2)
        ]

        _traj2 = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(block_ee_pos, goal_ee_pos, t, T//2)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, T//2),
            )
            for t in range(T//2)
        ]
        self._traj = _traj1 + _traj2
    
    def get_actions(self, env, xt, t):
        target_transform = self._traj[min(t, self._T - 1)]
        taus = []
        for env_idx in range(env._n_envs):
            ee_transform = env._franka.get_ee_transform(env_idx, env._franka_name)
        
            q_dot = env._franka.get_joints_velocity(env_idx, env._franka_name)[:7]
            J = env._franka.get_jacobian(env_idx, env._franka_name)
            x_vel = J @ q_dot

            tau, F = compute_task_space_impedance_control(J, ee_transform, target_transform, x_vel, self._Ks_0, self._Ds_0)
            taus.append(np.append(tau,F))
        taus = np.stack(taus, axis=0)
        return taus
    
    def apply_actions(self, env, actions):
        return env.step(actions[:, :7])[0]
    
    def execute(self, env, model, T_exec_max=300, plot=False, use_learned_model=False):
        env.init_joints = self._cfg_dict['initial_joints']
        x0 = env.reset()

        xf = self.generate_parameters(env)
        self.plan(model, env, x0, xf, 0, T_exec_max, plot=plot, use_learned_model=use_learned_model)
        tau_rollout = [np.zeros((T_exec_max, env.n+self.m))]*env._n_envs

        xt = x0.copy()
        for t in range(T_exec_max):
            actions = self.get_actions(env, xt, t)
            for env_idx in range(env._n_envs):
                tau_rollout[env_idx][t] = np.append(xt[env_idx], actions[env_idx])
                dist_to_block = np.linalg.norm(xt[env_idx][14:17] - self.init_block_pos[:3])
                if dist_to_block < self._thresh_dist:
                    env._franka.close_grippers(env_idx, env._franka_name)
            xt = self.apply_actions(env, actions)
        
        if plot:
            self.plot(tau_rollout[0], env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout', use_learned_model=use_learned_model, dof=3)
        return tau_rollout

    def plot(self, tau, env_name, goal, ctrl_name='iLQR_u_rollout', use_learned_model=False, dof=3):
        tau_gripper = np.concatenate((tau[:, 14:17], tau[:,21:24]), axis=1)
        control = tau[:, -6:-3]
        
        tau_block = np.concatenate((tau[:, 27:30], tau[:,34:37]), axis=1)
        fig, axes = plt.subplots(3, dof, figsize=(25, 15))
        
        for i in range(dof):
            axes[0, i].plot(tau_gripper[:,i], label=f'gripper pos{i}')
            axes[0, i].plot(tau_block[:,i], label=f'block pos{i}')
            axes[0, i].plot([goal[i]]*tau.shape[0], label=f'Goal{i}', linewidth=3)
            axes[0, i].set_xlabel('t', fontsize=20)
            axes[0, i].legend(prop={'size': 15})

            axes[1, i].plot(tau_gripper[:,i+dof], label=f'gripper velocity{i}')
            axes[1, i].plot(tau_block[:,i+dof], label=f'block velocity{i}')
            axes[1, i].set_xlabel('t', fontsize=20)
            axes[1, i].legend(prop={'size': 15})

            axes[2, i].plot(control[:,i], label=f'control{i}')
            axes[2, i].set_xlabel('t', fontsize=20)
            axes[2, i].legend(prop={'size': 15})
            
            if dof == 7:
                axes[0, i].set_ylabel('Theta', fontsize=20)
                axes[1, i].set_ylabel('Thetadot', fontsize=20)
                axes[2, i].set_ylabel('Torque', fontsize=20)
            else:
                axes[0, i].set_ylabel('Gripper Position', fontsize=20)
                axes[1, i].set_ylabel('Gripper Velocity', fontsize=20)
                axes[2, i].set_ylabel('Force', fontsize=20)
        
        model_name = 'plan_with_learned_model' if use_learned_model else 'plan_with_GT_model'
        plt.savefig(f'../Dynamic_GNN_structured_models/tests/media/{model_name}_{ctrl_name}_{env_name}.png')

class FrankaEEImpedanceControlDynamicSlidePickUp(FrankaEEImpedanceControlDynamicPickUp):
    def __init__(self, cfg_dict=None):
        super(FrankaEEImpedanceControlDynamicSlidePickUp, self).__init__(cfg_dict=cfg_dict)
        self._ctrl_name = 'Impedance_control_dynamic_slide_pickup'
        self._zpush = 0.00
        self._thresh_dist = 0.019

    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        self._T = T
        init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)

        self.init_block_pos = x0[0][27+self._cfg_dict['indx_block_to_pick']*13:27+(self._cfg_dict['indx_block_to_pick']+1)*13]

        goal_ee_transform = gymapi.Transform(
            p=gymapi.Vec3(xf[0], xf[1], xf[2]),
            r=init_ee_transform.r
        )
        env.goal_ee_transform = goal_ee_transform

        init_ee_pos = vec3_to_np(init_ee_transform.p)
        block_ee_pos = np.array(self.init_block_pos[:3])
        goal_ee_pos = vec3_to_np(goal_ee_transform.p)
        
        goal_ee_pos_mid = goal_ee_pos.copy()
        goal_ee_pos_mid[2] = block_ee_pos[2] - self._zpush
        goal_ee_pos_mid[:2] = (block_ee_pos[:2] + goal_ee_pos[:2])/2

        _traj1 = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(init_ee_pos, block_ee_pos, t, T//3)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, T//3),
            )
            for t in range(T//3)
        ]

        _traj2 = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(block_ee_pos, goal_ee_pos_mid, t, T//3)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, T//3),
            )
            for t in range(T//3)
        ]

        _traj3 = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(goal_ee_pos_mid, goal_ee_pos, t, T-2*T//3)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, T-2*T//3),
            )
            for t in range(T-2*T//3)
        ]

        self._traj = _traj1 + _traj2 + _traj3

class FrankaiLQROpenLoopJointSpace(Skill):
    def __init__(self, cfg_dict=None):
        super(FrankaiLQROpenLoopJointSpace, self).__init__(cfg_dict=cfg_dict)
        self._thresh_limit = 1e-6
        self._n_ilqr_iter = 20
        self._ctrl_name = 'Franka_iLQRjoint_openloop'
        self.m = 7
    
    def generate_parameters(self, env):
        # TODO: Goal sampling not happening here
        return np.append(np.array(self._cfg_dict['goal']['joints']), np.zeros(7))
    
    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        tau_init = env.initial_traj_interpolate(x0[0][:14], xf, T, model.n, model.m)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0[0][:14], xf)

        if t==0:
            env.intialize_cost_matrices(z0, zf, model.n, model.m, T)
            init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)
            goal_pos = self._cfg_dict['goal']['position']
            goal_ee_transform = gymapi.Transform(
                p=gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2]),
                r=init_ee_transform.r
            )
            env.goal_ee_transform = goal_ee_transform # Needed for plotting goal transform

        F, f = model.linearize_dynamics_obs_space(tau_init)
        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env)
    
        self.tau_pred = model._post_process_tau_batch_to_np(tau, model.n)
        self.control = control.copy()
    
    def get_actions(self, env, xt, t):
        return [self.tau_pred[t,-7:,0]]


class FrankaiLQROpenLoopCartesianSpace(Skill):
    def __init__(self, cfg_dict=None):
        super(FrankaiLQROpenLoopCartesianSpace, self).__init__(cfg_dict=cfg_dict)
        self._thresh_limit = 1e-6
        self._n_ilqr_iter = 20
        self._ctrl_name = 'Franka_iLQRcartesian_openloop'
        self.m = 6

        Kp_0, Kr_0 = 200, 8
        self._Ks_0 = np.diag([Kp_0] * 3 + [Kr_0] * 3)
        self._Ds_0 = np.diag([4 * np.sqrt(Kp_0)] * 3 + [2 * np.sqrt(Kr_0)] * 3)

        self._thresh_dist = 0.019

    def generate_parameters(self, env):
        if self._cfg_dict['goal']['randomize']:
            goal_pos = np.random.uniform(low=self._cfg_dict['goal']['position_ranges']['low'], high=self._cfg_dict['goal']['position_ranges']['high'])
        else:
            goal_pos = np.array(self._cfg_dict['goal']['position'])
        goal = np.tile(np.append(goal_pos, np.zeros(3)), env.N_O)
        return goal
    
    def plan(self, model, env, x0, xf, t, T, plot=False, use_learned_model=False):
        x0_cart = env.extract_cartesian_state(x0[0])
        tau_init = env.initial_traj_interpolate(x0_cart, xf, T, model.n, model.m)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0_cart, xf)
        if env._num_blocks > 0:
            self.init_block_pos = x0_cart[6:12].copy()
        if t==0:
            self.init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)
            goal_ee_transform = gymapi.Transform(
                        p=gymapi.Vec3(xf[0], xf[1], xf[2]),
                        r=self.init_ee_transform.r
                    )
            env.goal_ee_transform = goal_ee_transform
        
        if t==0:
            env.intialize_cost_matrices(z0, zf, model.n, model.m, T)

        F, f = model.linearize_dynamics_obs_space(tau_init)
        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters, self.mode_pred = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env, return_mode=self.return_mode)
        
        if self.return_mode:
            self.mode_pred = self.mode_pred.cpu().numpy()

        self.tau_pred = model._post_process_tau_batch_to_np(tau, model.n)
        self.control = control.copy()
    
    def compute_task_space_orientation_impedance_control(self, curr_transform, target_transform, r_vel, Ks, Ds):

        x_quat = quaternion.from_float_array(quat_to_np(curr_transform.r, format='wxyz'))
        xd_quat = quaternion.from_float_array(quat_to_np(target_transform.r, format='wxyz'))
        xe_ang_axis = angle_axis_between_quats(x_quat, xd_quat)

        F = -Ks @ xe_ang_axis - Ds @ r_vel
        return F

    def get_actions(self, env, xt, t):
        # ee_transform = env._franka.get_ee_transform(0, env._franka_name)
        # target_transform = gymapi.Transform(
        #             p=ee_transform.p,
        #             r=self.init_ee_transform.r
        #         )

        # J = env._franka.get_jacobian(0, env._franka_name)
        # q_dot = env._franka.get_joints_velocity(0, env._franka_name)[:7]
        # x_vel = J @ q_dot

        # F = self.compute_task_space_orientation_impedance_control(ee_transform, target_transform, x_vel[3:], self._Ks_0[3:,3:], self._Ds_0[3:,3:])
        # F_out = np.append(self.tau_pred[t,-3:,0], F)
        F_out = np.append(self.tau_pred[t,-3:,0], np.zeros(3))
        return [F_out]

    def apply_actions(self, env, actions):
        ee_transform = env._franka.get_ee_transform(0, env._franka_name)
        target_transform = gymapi.Transform(
                    p=ee_transform.p,
                    r=self.init_ee_transform.r
                )
        J = env._franka.get_jacobian(0, env._franka_name)
        q_dot = env._franka.get_joints_velocity(0, env._franka_name)[:7]
        x_vel = J @ q_dot

        F = self.compute_task_space_orientation_impedance_control(ee_transform, target_transform, x_vel[3:], self._Ks_0[3:,3:], self._Ds_0[3:,3:])

        actions[0][3:] = F.copy()
        tau = [ J.T@actions[0] ]
        return env.step(tau)[0]


class FrankaiLQROpenLoopCartesianSpaceReactive(FrankaiLQROpenLoopCartesianSpace):
    def __init__(self, cfg_dict=None):
        super(FrankaiLQROpenLoopCartesianSpaceReactive, self).__init__(cfg_dict=cfg_dict)
        self._ctrl_name = 'Franka_iLQRcartesian_reactive'
        self.reactive = False
        self.return_mode = False

        Kp_0, Kr_0 = 200, 8
        self._Ks_0 = np.diag([Kp_0] * 3 + [Kr_0] * 3)
        self._Ds_0 = np.diag([4 * np.sqrt(Kp_0)] * 3 + [2 * np.sqrt(Kr_0)] * 3)
    
    def plan(self, model, env, x0, h0, xf, t, T, plot=False, use_learned_model=False):
        x0_cart = env.extract_cartesian_state(x0[0])
        tau_init = env.initial_traj_interpolate(x0_cart, xf, T, model.n, model.m)
        dataz0, datazf, z0, zf = model.process_input_and_final_state(x0_cart, xf)
        if env._num_blocks > 0:
            self.init_block_pos = x0_cart[6:12].copy()
        if t==0:
            self.init_ee_transform = env._franka.get_ee_transform(0, env._franka_name)
            goal_ee_transform = gymapi.Transform(
                        p=gymapi.Vec3(xf[0], xf[1], xf[2]),
                        r=self.init_ee_transform.r
                    )
            env.goal_ee_transform = goal_ee_transform
        
        if t==0:
            env.intialize_cost_matrices(z0, zf, model.n, model.m, T)

        F, f = model.linearize_dynamics_obs_space(tau_init, h0, T)
        C, c = env.quadratize_cost(tau_init)
        tau, control, self.n_iters, self.mode_pred = iLQR1(dataz0, datazf, model.n*env.N_O, model.m, min(T-1,env.episode_len-t-1), F, f, C[t:t+T], c[t:t+T], 
                                            self._thresh_limit, self._n_ilqr_iter, model, env, h0=h0, return_mode=self.return_mode)
        
        if self.return_mode:
            self.mode_pred = self.mode_pred.cpu().numpy()

        self.tau_pred = model._post_process_tau_batch_to_np(tau, model.n)
        self.control = control.copy()

    def execute_mpc(self, env, model, T_exec_max=300, plot=False, use_learned_model=False):
        if use_learned_model:
            self.dof = model._cfg_dict['dof']
        self.N_O = env.N_O
        env.init_joints = self._cfg_dict['initial_joints']
        x0 = env.reset()
        xf = self.generate_parameters(env)

        tau_mpc = np.zeros((T_exec_max, env.n+self.m))
        tau_plan = []

        n_e = find_edge_index_pickup(self.N_O).shape[1]
        ht = torch.zeros(1, n_e, model._cfg_dict["num_edge_types"]).to(model._device)

        xt = x0.copy()
        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            # self.plan(model, env, xt, xf, t*self.rollout_deltat, T_exec_max-t*self.rollout_deltat, plot=False, use_learned_model=use_learned_model)
            self.plan(model, env, xt, ht, xf, t*self.rollout_deltat, self.t_plan, plot=False, use_learned_model=use_learned_model)
            tau_plan.append(self.tau_pred[:self.rollout_deltat])
            for t_roll in range(self.rollout_deltat):
                actions = self.get_actions(env, xt, t_roll)
                tau_mpc[t*self.rollout_deltat+t_roll] = np.append(xt[0], actions[0])
                if env._num_blocks > 0:
                    dist_to_block = np.linalg.norm(xt[0][14:17] - self.init_block_pos[:3])
                    # print(dist_to_block)
                    # all_ct_forces = env._scene.gym.get_rigid_contact_forces(env._scene.sim)
                    # all_ct = env._scene.gym.get_rigid_contacts(env._scene.sim)
                    # env._scene.gym.get_actor_index(env,env._franka,0)
                    # import ipdb; ipdb.set_trace()
                    if dist_to_block < self._thresh_dist:
                        env._franka.close_grippers(0, env._franka_name)
                xt = self.apply_actions(env, actions)
        tau_plan = np.concatenate(tau_plan, axis=0)
        if plot:
            self.plot(tau_mpc, env.name, xf, ctrl_name=self._ctrl_name+'_u_rollout_MPC', use_learned_model=use_learned_model, dof=self.dof)
            self.plot(tau_plan, env.name, xf, ctrl_name=self._ctrl_name+'_tau_pred_MPC', use_learned_model=use_learned_model, dof=self.dof)

        diff = self.find_prediction_error(env, tau_plan, tau_mpc)
        print(f"Prediction error={diff}")
        import ipdb; ipdb.set_trace()

        # Executing again
        x0 = env.reset()
        if env._num_blocks > 0:
            self.init_block_pos = x0[0][27:27+13]
        for t in trange(T_exec_max):
            xt = self.apply_actions(env, [tau_mpc[t,-self.m:]])
            if env._num_blocks > 0:
                dist_to_block = np.linalg.norm(xt[0][14:17] - self.init_block_pos[:3])
                # print(dist_to_block)
                if dist_to_block < self._thresh_dist:
                    env._franka.close_grippers(0, env._franka_name)
        return tau_mpc

class FrankaRolloutControl(Skill):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'RolloutControl'
        self.m = 3
        self.rollout_deltat = 50
    
    def execute(self, traj, model, T_exec_max=300, plot=False, use_learned_model=False):
        
        tau_rollout = np.zeros((traj.shape[0],15))
        self._num_blocks = 1
        tau_GT = np.zeros((traj.shape[0],15))
        for t in range(T_exec_max):
            tau_GT[t,:-3] = self.extract_cartesian_state(traj[t,:-6])
            tau_GT[t,-3:] = traj[t,-6:-3]

        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            x0 = self.extract_cartesian_state(traj[t*self.rollout_deltat,:-6])
            dataz0, _, z0, _ = model.process_input_and_final_state(x0, x0)
            control = traj[t*self.rollout_deltat:(t+1)*self.rollout_deltat, -6:-3]
            tau, _ = model.forward_propagate_control(dataz0, control)
            tau_rollout[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = model._post_process_tau_batch_to_np(tau, model.n)[:,:,0]
        if plot:
            self.plot(tau_rollout[:,:,None], 'none', x0, ctrl_name=self._ctrl_name+'_tau_rollout', use_learned_model=use_learned_model, dof=3)
            FrankaEEImpedanceControlDynamicPickUp.plot(FrankaEEImpedanceControlDynamicPickUp, traj, 'none', x0, ctrl_name=self._ctrl_name+'_tau_GT', use_learned_model=use_learned_model, dof=3)
        return tau_rollout, tau_GT
    
    def extract_cartesian_state(self, state):
        state_cart = np.append(state[14:17], state[21:24])
        for i in range(0, self._num_blocks):
            state_cart = np.append(
                state_cart,
                np.append(state[27+13*i:27+13*i+3], state[27+13*i+7:27+13*i+10]) 
                )
        return state_cart

class FrankaRolloutControlReactive(Skill):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'RolloutControlReactive'
        self.m = 3
        self.rollout_deltat = 50
    
    def execute(self, traj, model, T_exec_max=300, plot=False, use_learned_model=False):
        
        n_e = 2
        ht = torch.zeros(1, n_e, model._cfg_dict["num_edge_types"]).to(model._device)
        
        tau_rollout = np.zeros((traj.shape[0],15))
        self._num_blocks = 1
        tau_GT = np.zeros((traj.shape[0],15))
        for t in range(T_exec_max):
            tau_GT[t,:-3] = self.extract_cartesian_state(traj[t,:-6])
            tau_GT[t,-3:] = traj[t,-6:-3]

        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            x0 = self.extract_cartesian_state(traj[t*self.rollout_deltat,:-6])
            dataz0, _, z0, _ = model.process_input_and_final_state(x0, x0)
            if t>0:
                ht = model.update_ht_fwd_prop(ht, dataz0)
            control = traj[t*self.rollout_deltat:(t+1)*self.rollout_deltat, -6:-3]
            tau, ht = model.forward_propagate_control(dataz0, control, ht=ht)
            tau_rollout[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = model._post_process_tau_batch_to_np(tau, model.n)[:,:,0]

        if plot:
            self.plot(tau_rollout[:,:,None], 'none', x0, ctrl_name=self._ctrl_name+'_tau_rollout', use_learned_model=use_learned_model, dof=3)
            FrankaEEImpedanceControlDynamicPickUp.plot(FrankaEEImpedanceControlDynamicPickUp, traj, 'none', x0, ctrl_name=self._ctrl_name+'_tau_GT', use_learned_model=use_learned_model, dof=3)

        return tau_rollout, tau_GT
    
    def extract_cartesian_state(self, state):
        state_cart = np.append(state[14:17], state[21:24])
        for i in range(0, self._num_blocks):
            state_cart = np.append(
                state_cart,
                np.append(state[27+13*i:27+13*i+3], state[27+13*i+7:27+13*i+10]) 
                )
        return state_cart

class RealFrankaRolloutControl(Skill):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'RolloutControl'
        self.m = 3
        self.rollout_deltat = 500
    
    def execute(self, traj, model, T_exec_max=300, plot=False, use_learned_model=False):
        
        tau_rollout = np.zeros((traj.shape[0],15))
        self._num_blocks = 1

        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            x0 = traj[t*self.rollout_deltat,:-3]
            dataz0, _, z0, _ = model.process_input_and_final_state(x0, x0)
            control = traj[t*self.rollout_deltat:(t+1)*self.rollout_deltat, -3:]
            tau, _ = model.forward_propagate_control(dataz0, control)
            tau_rollout[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = model._post_process_tau_batch_to_np(tau, model.n)[:,:,0]
        if plot:
            self.plot(tau_rollout[:,:,None], 'RealFranka', x0, ctrl_name=self._ctrl_name+'_', use_learned_model=use_learned_model, dof=3)
            self.plot(traj[:,:,None], 'RealFranka', x0, ctrl_name=self._ctrl_name+'_', use_learned_model=use_learned_model, dof=3)
        return tau_rollout, traj


class RealFrankaRolloutControlReactive(Skill):
    def __init__(self):
        super().__init__()
        self._ctrl_name = 'RolloutControlReactive'
        self.m = 3
        self.rollout_deltat = 500
    
    def execute(self, traj, model, T_exec_max=300, plot=False, use_learned_model=False):
        
        n_e = 2
        ht = torch.zeros(1, n_e, model._cfg_dict["num_edge_types"]).to(model._device)
        
        tau_rollout = np.zeros((traj.shape[0],15))
        self._num_blocks = 1

        for t in trange(int(T_exec_max/self.rollout_deltat), desc="MPC steps"):
            x0 = traj[t*self.rollout_deltat,:-3]
            dataz0, _, z0, _ = model.process_input_and_final_state(x0, x0)
            if t>0:
                ht = model.update_ht_fwd_prop(ht, dataz0)
            control = traj[t*self.rollout_deltat:(t+1)*self.rollout_deltat, -3:]
            tau, ht = model.forward_propagate_control(dataz0, control, ht=ht)
            tau_rollout[t*self.rollout_deltat:(t+1)*self.rollout_deltat] = model._post_process_tau_batch_to_np(tau, model.n)[:,:,0]

        if plot:
            self.plot(tau_rollout[:,:,None], 'RealFranka', x0, ctrl_name=self._ctrl_name+'_', use_learned_model=use_learned_model, dof=3)
            self.plot(traj[:,:,None], 'RealFranka', x0, ctrl_name=self._ctrl_name+'_', use_learned_model=use_learned_model, dof=3)

        return tau_rollout, traj