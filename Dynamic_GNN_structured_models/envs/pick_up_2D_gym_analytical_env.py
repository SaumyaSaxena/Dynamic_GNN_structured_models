import numpy as np
import gym
from scipy.special import softmax

class PickUp2DGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, cfg):
        self._cfg = cfg
        self.name = 'PickUp2DGymEnv'
        self.episode_len = cfg['gym']['episode_len']
        self.n = 4 # size per object
        self.m = 2
        self.N_O = 2
        self.force_mag = 1.0

        self.goal_pos = cfg['gym']['goal']['position']
        self.goal_vel = cfg['gym']['goal']['velocity']
        self.goal_randomize = cfg['gym']['goal']['randomize']
        self.goal_pos_ranges = cfg['gym']['goal']['position_ranges']
        self.goal_vel_ranges = cfg['gym']['goal']['velocity_ranges']

        dt = cfg['scene']['dt']

        mass_g = cfg['env_props']['dynamics']['gripper']['mass']
        mass_o = cfg['env_props']['dynamics']['block']['mass']

        self._radius_g = cfg['env_props']['dynamics']['gripper']['dim']
        self._radius_o = cfg['env_props']['dynamics']['block']['dim']

        self._w_class = np.array([[-(self._radius_g+self._radius_o), 1],[(self._radius_g+self._radius_o), -1]])

        Apart = np.array(([1., 0., dt, 0.],
                        [0., 1., 0., dt],
                        [0., 0., 1., 0],
                        [0., 0., 0., 1]))

        self._A = np.zeros((self.n*self.N_O, self.n*self.N_O))
        for i in range(self.N_O):
            self._A[i*self.n:(i+1)*self.n, i*self.n:(i+1)*self.n] = Apart
            
        self._B1 = np.array(([0., 0.], 
                        [0., 0.], 
                        [dt/mass_g, 0.],
                        [0., dt/mass_g], 
                        [0., 0.], 
                        [0., 0.], 
                        [0., 0.],
                        [0., 0.]))
        self._B2 = np.array(([0., 0.], 
                        [0., 0.], 
                        [dt/(mass_g+mass_o), 0.],
                        [0., dt/(mass_g+mass_o)], 
                        [0., 0.], 
                        [0., 0.], 
                        [dt/(mass_g+mass_o), 0.],
                        [0., dt/(mass_g+mass_o)]))
        self._C = np.array(([1, 0., 0., 0., 0., 0., 0., 0.], 
                            [0., 1., 0., 0., 0., 0., 0., 0.], 
                            [0., 0., mass_g/(mass_g+mass_o), 0., 0., 0., mass_o/(mass_g+mass_o), 0.],
                            [0., 0., 0., mass_g/(mass_g+mass_o), 0., 0., 0., mass_o/(mass_g+mass_o)], 
                            [0., 0., 0., 0., 1., 0., 0., 0.], 
                            [0., 0., 0., 0., 0., 1., 0., 0.], 
                            [0., 0., mass_g/(mass_g+mass_o), 0., 0., 0., mass_o/(mass_g+mass_o), 0.],
                            [0., 0., 0., mass_g/(mass_g+mass_o), 0., 0., 0., mass_o/(mass_g+mass_o)]))
        
        if cfg['gym']['usage']['data_collection']:
            self.observation_space = gym.spaces.Box(-np.inf*np.ones(self.n*self.N_O), 
                                                    np.inf*np.ones(self.n*self.N_O), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.m,))
        self.reset()

    def _set_initial_states(self):
        self._gripper_pos = np.array(self._cfg['env_props']['initial_poses']['gripper']['position'])
        self._gripper_vel = np.array(self._cfg['env_props']['initial_poses']['gripper']['velocity'])
        self._block_pos = np.array(self._cfg['env_props']['initial_poses']['block']['position'])
        self._block_vel = np.array(self._cfg['env_props']['initial_poses']['block']['velocity'])

    def reset(self):
        self._current_timestep = 0
        self._set_initial_states()
        randomize = self._cfg['env_props']['initial_poses']['gripper']['randomize'] or self._cfg['env_props']['initial_poses']['block']['randomize']
        potential_gripper_pos = self._gripper_pos.copy()
        potential_block_pos = self._block_pos.copy()
        if randomize:
            for sample in range(self._cfg['env_props']['initial_poses']['max_samples']):
                if self._cfg['env_props']['initial_poses']['gripper']['randomize']:
                    pose_ranges = self._cfg['env_props']['initial_poses']['gripper']['position_ranges']
                    potential_gripper_pos = np.random.uniform(low=pose_ranges['low'], high=pose_ranges['high'])
                    velocity_ranges = self._cfg['env_props']['initial_poses']['gripper']['velocity_ranges']
                    self._gripper_vel = np.random.uniform(low=velocity_ranges['low'], high=velocity_ranges['high'])
                if self._cfg['env_props']['initial_poses']['block']['randomize']:
                    pose_ranges = self._cfg['env_props']['initial_poses']['block']['position_ranges']
                    potential_block_pos = np.random.uniform(low=pose_ranges['low'], high=pose_ranges['high'])
                    velocity_ranges = self._cfg['env_props']['initial_poses']['block']['velocity_ranges']
                    self._block_vel = np.random.uniform(low=velocity_ranges['low'], high=velocity_ranges['high'])
                if not self._is_in_collision(potential_gripper_pos, potential_block_pos):
                    self._gripper_pos = potential_gripper_pos.copy()
                    self._block_pos = potential_block_pos.copy()
                    break

        self._modetm1 = 0
        self._modet = self._find_mode(self._gripper_pos, self._block_pos)
        return np.concatenate([self._gripper_pos, self._gripper_vel, self._block_pos, self._block_vel])

    def _is_in_collision(self, gripper_pos, block_pos):
        dist = np.linalg.norm(gripper_pos - block_pos)
        return dist < (self._radius_g+self._radius_o)

    def step(self, action):
        self._current_timestep += 1
        ut = np.array([action[0]*self.force_mag, action[1]*self.force_mag])
        xt = np.concatenate([self._gripper_pos, self._gripper_vel, self._block_pos, self._block_vel])[:,None]

        if self._modet == self._modetm1:
            self.At = self._A.copy()
            if self._modet == 0:
                xtp1 = self._A@xt + self._B1@ut
                self.Bt = self._B1.copy()
            else:
                xtp1 = self._A@xt + self._B2@ut
                self.Bt = self._B2.copy()
        else:
            self.At = self._A@self._C
            if self._modet == 0:
                xtp1 = self._A@self._C@xt + self._B1@ut
                self.Bt = self._B1.copy()
            else:
                xtp1 = self._A@self._C@xt + self._B2@ut
                self.Bt = self._B2.copy()

        modetp1 = self._find_mode(xtp1[:int(self.n/2)], xtp1[self.n : self.n+int(self.n/2)])

        self._gripper_pos = xtp1[:int(self.n/2),0]
        self._gripper_vel = xtp1[int(self.n/2):self.n,0]
        self._block_pos = xtp1[self.n:self.n+int(self.n/2),0]
        self._block_vel = xtp1[self.n+int(self.n/2):2*self.n,0]
        self._modetm1 = self._modet + 0
        self._modet = modetp1 + 0
        
        done = True if self._current_timestep > self.episode_len else False
        reward = 0
        return xtp1, reward, done, {}

    def _find_mode(self, gripper_pos, block_pos):
        mode =  np.argmax(softmax(
            self._w_class@np.append(1, np.linalg.norm(gripper_pos - block_pos))
            ))
        return mode

    def linearize_dynamics(self, tau, *args):
        # tau has shape (T+1, n * n_objects + m, 1)
        F = np.zeros((tau.shape[0], self.n*self.N_O, self.n*self.N_O + self.m))
        f = np.zeros((tau.shape[0], self.n*self.N_O, 1))
        Cost = np.zeros((tau.shape[0], self.n*self.N_O + self.m, self.n*self.N_O + self.m))
        c = np.zeros((tau.shape[0], self.n*self.N_O + self.m, 1))
        mode = np.zeros(tau.shape[0])

        mode[0] = self._find_mode(tau[0,:int(self.n/2),:], tau[0,self.n:self.n+int(self.n/2),:])
        F[0, :, :self.n*self.N_O] = self._A
        F[0, :, self.n*self.N_O:] = self._B1

        for i in range(1, tau.shape[0]):
            mode[i] = self._find_mode(tau[i,:int(self.n/2),:], tau[i,self.n:self.n+int(self.n/2),:])
            F[i, :, :self.n*self.N_O] = self._A if mode[i-1] == mode[i] else self._A@self._C
            F[i, :, self.n*self.N_O:] = self._B1 if mode[i]==0 else self._B2

        return F, f, self.C, self.c

    def linearize_dynamics_obs_space(self, tau):
        F, f, Cost, c = self.linearize_dynamics(tau)
        return F, f

    def process_input_and_final_state(self, x0, xf):
        return x0.reshape(self.n*self.N_O,1), xf.reshape(self.n*self.N_O,1), x0.reshape(self.n*self.N_O,1), xf.reshape(self.n*self.N_O,1)
    
    def initial_traj_interpolate(self, x0, xf, T):
        xf1 = np.zeros(x0.flatten().shape[0])
        
        xf1[:int(self.n/2)] = x0.flatten()[self.n:self.n+int(self.n/2)]
        xf1[self.n:self.n+int(self.n/2)] = x0.flatten()[self.n:self.n+int(self.n/2)]

        # xf1 = xf/2

        tau = np.append(np.linspace(x0.flatten(), xf1, int(T/2)),
                        np.linspace(xf1, xf.flatten(), T-int(T/2)), axis=0)
        tau = np.append(tau, np.zeros((T, self.m)), axis=1)[:,:,None]
        return tau
    
    def intialize_cost_matrices(self, zf2, n, pick_up_seq):
        block_cost = np.diag([1.5,1.5,10.,10.])
        Q1 = np.append( 
                (np.append(block_cost, -1*block_cost, axis=1)), 
                (np.append(-1*block_cost, block_cost, axis=1)), axis=0) # distance between two objects

        Q2 = 1.0*np.diag([1e-6]*n+[1e-6]*n) # distance to goal
        
        R1 = 0.1*np.eye(self.m)
        R2 = 0.1*np.eye(self.m)
        QR = np.zeros((n*self.N_O, self.m))
        RQ = np.zeros((self.m, n*self.N_O))

        QQ = Q1 + Q2
        C1 = np.append( (np.append(QQ, QR, axis=1)), (np.append(RQ, R1, axis=1)), axis=0)
        C2 = np.append( (np.append(QQ, QR, axis=1)), (np.append(RQ, R2, axis=1)), axis=0)

        cx1 = np.zeros((n*self.N_O,1))
        cx2 = -Q2@zf2
        cxx = cx1 + cx2

        cu = np.zeros((self.m, 1))
        c1 = np.append(cxx, cu, axis=0)

        self.C = np.repeat(C1[np.newaxis, :, :], self.episode_len, axis=0)
        self.c = np.repeat(c1[np.newaxis, :, :], self.episode_len, axis=0)
        const = zf2.reshape(1, n*self.N_O)@(QQ)@zf2.reshape(n*self.N_O, 1)
        self.const = np.repeat(const[np.newaxis, :], self.episode_len, axis=0)

    def quadratize_cost(self, tau):
        return self.C, self.c

    def forward_propagate_control_lqr(self, x0, control, *args, **kwargs):
        self._gripper_pos = x0[:int(self.n/2),0].copy()
        self._gripper_vel = x0[int(self.n/2):self.n,0].copy()
        self._block_pos = x0[self.n:self.n+int(self.n/2),0].copy()
        self._block_vel = x0[self.n+int(self.n/2):,0].copy()
        self._modetm1 = 0
        self._modet = self._find_mode(self._gripper_pos, self._block_pos)

        T = control['K'].shape[0]

        tau = np.zeros((T, self.n*self.N_O+self.m, 1))
        F = np.zeros((T, self.n*self.N_O, self.n*self.N_O + self.m))
        f = np.zeros((T, self.n*self.N_O, 1))

        xt = np.concatenate([self._gripper_pos, self._gripper_vel, self._block_pos, self._block_vel])[:,None]
        for t in range(T):
            ut = control['K'][t]@xt + control['k'][t]
            tau[t] = np.append(xt, ut, axis = 0)
            xt, *_ = self.step(ut)
            F[t] = np.append(self.At, self.Bt, axis=1)
        
        self._gripper_pos = x0[:int(self.n/2),0].copy()
        self._gripper_vel = x0[int(self.n/2):self.n,0].copy()
        self._block_pos = x0[self.n:self.n+int(self.n/2),0].copy()
        self._block_vel = x0[self.n+int(self.n/2):,0].copy()
        self._modetm1 = 0
        self._modet = self._find_mode(self._gripper_pos, self._block_pos)
        return tau, F, f, None

    def post_process_tau_latent_to_obs(self, tau, n, is_numpy=True):
        return tau

    def close(self):
        pass
        
    def render(self):
        pass
