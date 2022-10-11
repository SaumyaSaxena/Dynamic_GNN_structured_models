import numpy as np
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.math_utils import transform_to_np, rpy_to_quat
from isaacgym_utils.draw import draw_transforms

class FrankaPickupIsaacgymEnv():
    def __init__(self, cfg):
        self._cfg = cfg
        self.init_joints = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04])
        self.name = 'FrankaPickupIsaacgymEnv'
        self._n_envs = cfg['scene']['n_envs']
        self._scene = GymScene(cfg['scene'])

        self._table = GymBoxAsset(self._scene, **cfg['table']['dims'], 
                            shape_props=cfg['table']['shape_props'], 
                            asset_options=cfg['table']['asset_options']
                            )
        self._franka = GymFranka(cfg['franka'], self._scene, actuation_mode='torques')

        self._block = GymBoxAsset(self._scene, **cfg['block']['dims'], 
                            shape_props=cfg['block']['shape_props'], 
                            rb_props=cfg['block']['rb_props'],
                            asset_options=cfg['block']['asset_options']
                            )
        
        self._collision_eps = 2e-3

        table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
        franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))

        self._num_blocks = len(cfg['block']['initial_poses']['positions'])
        block_poses = cfg['block']['initial_poses']['positions']
        block_transforms = [
                    gymapi.Transform(p=gymapi.Vec3(pose[0], pose[1], cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + self._collision_eps),
                                    r=rpy_to_quat([0, 0, 0]))
                    for pose in block_poses]

        self._franka_name = 'franka'
        self._table_name = 'table'
        self._block_names = [f'block{i}' for i in range(len(block_transforms))]

        self.N_O = 1+self._num_blocks
        def setup(scene, _):
            collision_filter = 1
            self._scene.add_asset(self._table_name, self._table, table_transform, collision_filter=collision_filter)
            collision_filter *= 2
            self._scene.add_asset(self._franka_name, self._franka, franka_transform, collision_filter=collision_filter) # avoid self-collisions
            for block_name, block_transform in zip(self._block_names, block_transforms):
                collision_filter *= 2
                self._scene.add_asset(block_name, self._block, block_transform, collision_filter=collision_filter)
        self._scene.setup_all_envs(setup)

        self.goal_ee_transform = None

        self.episode_len = cfg['gym']['episode_len']
        self.n = 27 + self._num_blocks*13
        self.m = 7
        self.reset()
    
    def reset(self):
        self._current_timestep = 0

        if self._cfg['block']['initial_poses']['randomize']:
            position_ranges = self._cfg['block']['initial_poses']['position_ranges']
            block_poses = [np.random.uniform(low=position_ranges['low'], high=position_ranges['high'])
                            for _ in range(self._num_blocks)]
        else:
            block_poses = self._cfg['block']['initial_poses']['positions']

        block_transforms = [
                    gymapi.Transform(p=gymapi.Vec3(pose[0], pose[1], self._cfg['table']['dims']['sz'] + self._cfg['block']['dims']['sz'] / 2 + self._collision_eps),
                                    r=rpy_to_quat([0, 0, 0]))
                    for pose in block_poses]
        
        
        for env_idx in range(self._n_envs):
            self._franka.set_joints(env_idx, self._franka_name, self.init_joints)
            self._franka.set_joints_targets(env_idx, self._franka_name, np.array(self.init_joints))

            for block_name, block_transform in zip(self._block_names, block_transforms):
                self._block.set_rb_transforms(env_idx, block_name, [block_transform])
        for i in range(10):
            self._scene.step()
            self._scene.render(custom_draws=self.custom_draws)
        return self._get_current_state()

    def step(self, action):
        for env_idx in range(self._n_envs):
            self._franka.apply_torque(env_idx, self._franka_name, action[env_idx])

        self._scene.step()
        self._scene.render(custom_draws=self.custom_draws)
        self._current_timestep += 1

        next_state = self._get_current_state()

        done = True if self._current_timestep > self.episode_len else False
        reward = 0

        return next_state, reward, done, {}
    
    def _get_current_state(self):
        states = []
        for env_idx in range(self._n_envs):
            joints = self._franka.get_joints(env_idx, self._franka_name)[:7]
            joints_velocity = self._franka.get_joints_velocity(env_idx, self._franka_name)[:7]

            ee_transform_np = transform_to_np(self._franka.get_ee_transform(env_idx, self._franka_name), format='wxyz')
            ee_position = ee_transform_np[:3]
            ee_quaternion = ee_transform_np[3:]

            ee_vels = self._franka.get_jacobian(env_idx, self._franka_name) @ joints_velocity
            ee_linear_velocity = ee_vels[:3]
            ee_angular_velocity = ee_vels[3:]
            
            state = np.concatenate((joints, joints_velocity, ee_position, ee_quaternion, ee_linear_velocity, ee_angular_velocity))

            for block_name in self._block_names:
                block_pose = self._block.get_rb_poses_as_np_array(env_idx, block_name)[0]
                block_vel = self._block.get_rb_vels_as_np_array(env_idx, block_name)[0]
                state = np.concatenate((state, block_pose[:3], block_pose[3:], block_vel[:3][0], block_vel[:3][1]))
            states.append(state)
        return states
    
    def extract_cartesian_state(self, state):
        state_cart = np.append(state[14:17], state[21:24])
        for i in range(0, self._num_blocks):
            state_cart = np.append(
                state_cart,
                np.append(state[27+13*i:27+13*i+3], state[27+13*i+7:27+13*i+10]) 
                )
        return state_cart
    
    def custom_draws(self, scene):
        for env_idx in scene.env_idxs:
            franka_ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)
            draw_transforms(scene, [env_idx], [franka_ee_transform, self.goal_ee_transform])

    def initial_traj_interpolate(self, x0, xf, T, n, m):
        if self._num_blocks == 0:
            tau = np.linspace(x0.flatten(), x0.flatten(), T)
            tau = np.append(tau, np.zeros((T, m)), axis=1)[:,:,None]
        else:
            # Always picking up first block. Order doesn't matter with GNNs
            xf1 = np.tile(x0.flatten()[n:2*n], self.N_O)
            tau = np.append(np.linspace(x0.flatten(), xf1, int(T/2)),
                            np.linspace(xf1, xf.flatten(), T-int(T/2)), axis=0)
            tau = np.append(tau, np.zeros((T, m)), axis=1)[:,:,None]
        return tau
    
    def intialize_cost_matrices(self, x0, xf, n, m, T):
        if self._num_blocks == 0:
            Q = 1e+3*np.eye(n)
            R = 1e-0*np.eye(m)
            QR = np.zeros((n, m))
            RQ = np.zeros((m, n))

            C = np.append( np.append(Q, QR, axis=1), np.append(RQ, R, axis=1), axis=0)

            cx = -Q@xf
            cu = np.zeros((m, 1))
            c = np.append(cx, cu, axis=0)

            self.C = np.repeat(C[np.newaxis, :, :], T, axis=0)
            self.c = np.repeat(c[np.newaxis, :, :], T, axis=0)
        else:
            # Always picking up first block. Order doesn't matter with GNNs
            Q1 = np.tile(1e-6*np.eye(n), (self.N_O, self.N_O))
            block_cost = np.diag([1e+3,2.8e+2,3e+2,5e-0,1e-1,2e-1])
            Q1[:n,:n] = 1.0*block_cost
            Q1[n:2*n,:n] = -1.0*block_cost
            Q1[:n,n:2*n] = -1.0*block_cost
            Q1[n:2*n,n:2*n] = 1.0*block_cost

            Q2 = 1e-6*np.eye(n*self.N_O) # distance to goal
            block_cost2 = np.diag([5e+3,3e+3,5e+3,1e+0,9e+1,1e-1])
            Q2[n:2*n,n:2*n] = 1.0*block_cost2
            
            R = 1e-2*np.eye(m)
            QR = np.zeros((n*self.N_O, m))
            RQ = np.zeros((m, n*self.N_O))

            Q = Q1 + Q2
            C = np.append( np.append(Q, QR, axis=1), np.append(RQ, R, axis=1), axis=0)
            
            cx1 = np.zeros((n*self.N_O,1))
            cx2 = -Q2@xf
            
            cx = cx1 + cx2
            cu = np.zeros((m, 1))
            c = np.append(cx, cu, axis=0)

            self.C = np.repeat(C[np.newaxis, :, :], self.episode_len, axis=0)
            self.c = np.repeat(c[np.newaxis, :, :], self.episode_len, axis=0)

    def quadratize_cost(self, tau):
        return self.C, self.c