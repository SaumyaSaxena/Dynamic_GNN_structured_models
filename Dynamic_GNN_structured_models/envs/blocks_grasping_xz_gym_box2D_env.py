import gym
from gym import error, spaces, utils
from gym.utils import colorize, seeding, EzPickle
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, circleShape, revoluteJointDef, contactListener)
import numpy as np
import ipdb

FPS = 20 #60

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        for i in range(self.env._num_blocks):
            if (self.env._blocks[i]==contact.fixtureB.body) or (self.env._blocks[i]==contact.fixtureA.body):
                self.env._grasps[i] = True

    def EndContact(self, contact):
        pass
        
class BlocksGraspXZ(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    def __init__(self, cfg):
        EzPickle.__init__(self)
        self._cfg = cfg
        self.name = 'BlocksGraspXZ'
        self.seed()
        self.viewer = None
        self.force_mag = 1.0
        self._collision_thresh = 5e-2
        
        self._dt = cfg['scene']['dt']
        self._fps = 1.0 // self._dt

        self.world = Box2D.b2World()
        self.world.gravity = (cfg['scene']['gravity'][0], cfg['scene']['gravity'][1])
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround

        self._num_blocks = len(cfg['env_props']['initial_poses']['blocks']['positions'])
        self.N_O = 1+self._num_blocks
        self.n = 4 # state len per object
        self.m = 2 # control len
        self.episode_len = cfg['gym']['episode_len']

        if cfg['env_props']['dynamics']['gripper']['shape'] == 'circle':
            self._radius_G = cfg['env_props']['dynamics']['gripper']['dim']
            density_G = cfg['env_props']['dynamics']['gripper']['mass']/(np.pi*self._radius_G*self._radius_G)
            BOX_FD_G = fixtureDef(
                            shape=circleShape(radius=self._radius_G),
                            density=density_G,
                            friction=cfg['env_props']['dynamics']['gripper']['friction'],
                            restitution=cfg['env_props']['dynamics']['gripper']['restitution'])
        elif cfg['env_props']['dynamics']['gripper']['shape'] == 'box':
            self._width_G = cfg['env_props']['dynamics']['gripper']['dim']
            density_G = cfg['env_props']['dynamics']['gripper']['mass']/(self._width_G**2)
            BOX_FD_G = fixtureDef(
                            shape=polygonShape(box=(self._width_G/2, self._width_G/2)),
                            density=density_G,
                            friction=cfg['env_props']['dynamics']['gripper']['friction'],
                            restitution=cfg['env_props']['dynamics']['gripper']['restitution'])
        self._gripper = self.world.CreateDynamicBody(
                            position = (cfg['env_props']['initial_poses']['gripper']['position'][0], 
                                        cfg['env_props']['initial_poses']['gripper']['position'][1]),
                            angle = (0),
                            fixtures = BOX_FD_G,
                            fixedRotation = True
                            )
        self._gripper.color1 = (0.0,0.0,0)
        self._gripper.color2 = (0,0,0)

        if cfg['env_props']['dynamics']['blocks']['shape'] == 'circle':
            self._radius_B = cfg['env_props']['dynamics']['blocks']['dim']
            density_B = cfg['env_props']['dynamics']['blocks']['mass']/(np.pi*self._radius_B*self._radius_B)
            BOX_FD_B = fixtureDef(
                            shape=circleShape(radius=self._radius_B),
                            density=density_B,
                            friction=cfg['env_props']['dynamics']['blocks']['friction'],
                            restitution=cfg['env_props']['dynamics']['blocks']['restitution'])
        elif cfg['env_props']['dynamics']['blocks']['shape'] == 'box':
            self._width_B = cfg['env_props']['dynamics']['blocks']['dim']
            density_B = cfg['env_props']['dynamics']['blocks']['mass']/(self._width_B**2)
            BOX_FD_B = fixtureDef(
                            shape=polygonShape(box=(self._width_B/2, self._width_B/2)),
                            density=density_B,
                            friction=cfg['env_props']['dynamics']['blocks']['friction'],
                            restitution=cfg['env_props']['dynamics']['blocks']['restitution'])
        block_positions = cfg['env_props']['initial_poses']['blocks']['positions']
        self._blocks = [self.world.CreateDynamicBody(
                            position = (position[0], 
                                        position[1]),
                            angle = (0),
                            fixtures = BOX_FD_B,
                            fixedRotation = True
                            ) for position in block_positions]
        for block in self._blocks:
            block.color1 = (0.644, 1.0, 0.0)
            block.color2 = (0,0,0)

        self._graspJoints = [None]*self._num_blocks

        self._drawlist = [self._gripper] + self._blocks

        self.goal_pos = cfg['gym']['goal']['position']
        self.goal_vel = cfg['gym']['goal']['velocity']
        self.goal_randomize = cfg['gym']['goal']['randomize']
        self.goal_pos_ranges = cfg['gym']['goal']['position_ranges']
        self.goal_vel_ranges = cfg['gym']['goal']['velocity_ranges']

        high = 10
        self.action_space = spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-1.5, -1.5, -high, -high, -high, -high, -high, -high,-high, -high, -high, -high,-2]), np.array([1.5, 1.5, high, high, high, high, high, high, high, high, high, high,2]), dtype=np.float32)
        
        self.reset()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self._gripper: return
        self.world.DestroyBody(self._gripper)
        self._gripper = None
        for i in range(self._num_blocks):
            self.world.DestroyBody(self._blocks[i])
            self.world.DestroyJoint(self._graspJoints[i])
        self._blocks = None

    def reset_to_state(self, gripper_state, block_state):
        for i in range(self._num_blocks):
            if self._graspJoints[i] is not None:
                self.world.DestroyJoint(self._graspJoints[i])

        self._gripper.position = (gripper_state[0], gripper_state[1])
        self._gripper.linearVelocity = (gripper_state[2], gripper_state[3])
        for i in range(self._num_blocks):
            self._blocks[i].position = (block_state[i,0], block_state[i,1])
            self._blocks[i].linearVelocity = (block_state[i,2], block_state[i,3])

        self._prev_grasps = [False]*self._num_blocks
        self._grasps = [False]*self._num_blocks
        self._graspJoints = [None]*self._num_blocks

    def reset(self):
        self._current_timestep = 0

        potential_gripper_pos = np.array(self._cfg['env_props']['initial_poses']['gripper']['position'])
        gripper_pos = potential_gripper_pos.copy()
        gripper_vel = np.array(self._cfg['env_props']['initial_poses']['gripper']['velocity'])
        potential_block_pos = [np.array(self._cfg['env_props']['initial_poses']['blocks']['positions'][i])
                            for i in range(self._num_blocks)]
        block_pos = potential_block_pos.copy()
        block_vel = [np.array(self._cfg['env_props']['initial_poses']['blocks']['velocities'][i])
                            for i in range(self._num_blocks)]

        randomize = self._cfg['env_props']['initial_poses']['gripper']['randomize'] or self._cfg['env_props']['initial_poses']['blocks']['randomize']
        if randomize:
            for sample in range(self._cfg['env_props']['initial_poses']['max_samples']):
                if self._cfg['env_props']['initial_poses']['gripper']['randomize']:
                    pose_ranges = self._cfg['env_props']['initial_poses']['gripper']['position_ranges']
                    potential_gripper_pos = np.random.uniform(low=pose_ranges['low'], high=pose_ranges['high'])
                    velocity_ranges = self._cfg['env_props']['initial_poses']['gripper']['velocity_ranges']
                    gripper_vel = np.random.uniform(low=velocity_ranges['low'], high=velocity_ranges['high'])
                if self._cfg['env_props']['initial_poses']['blocks']['randomize']:
                    pose_ranges = self._cfg['env_props']['initial_poses']['blocks']['position_ranges']
                    velocity_ranges = self._cfg['env_props']['initial_poses']['blocks']['velocity_ranges']
                    potential_block_pos = [np.random.uniform(low=pose_ranges['low'], high=pose_ranges['high'])
                                            for _ in range(self._num_blocks)]
                    block_vel = [np.random.uniform(low=velocity_ranges['low'], high=velocity_ranges['high'])
                                        for _ in range(self._num_blocks)]
                if not self._is_in_collision(potential_gripper_pos, potential_block_pos):
                    gripper_pos = potential_gripper_pos.copy()
                    block_pos = potential_block_pos.copy()
                    break 
        
        self.reset_to_state(np.append(gripper_pos, gripper_vel), np.append(np.stack(block_pos,axis=0), np.stack(block_vel,axis=0), axis=1))
        
        return self._get_state().squeeze()

    def _is_in_collision(self, gripper_pos, block_pos):
        # TODO: Implement block to block collision check
        for i in range(len(block_pos)):
            dist = np.linalg.norm(gripper_pos - block_pos[i])
            if self._cfg['env_props']['dynamics']['blocks']['shape'] == 'circle':
                if dist < (self._radius_G+self._radius_B)+self._collision_thresh:
                    return True
            elif self._cfg['env_props']['dynamics']['blocks']['shape'] == 'box':
                if dist < (self._width_G+self._width_B)/np.sqrt(2)+self._collision_thresh:
                    return True
        return False

    def _get_state(self):
        state = [
            self._gripper.position[0],
            self._gripper.position[1],
            self._gripper.linearVelocity[0],
            self._gripper.linearVelocity[1]
            ]
        for i in range(self._num_blocks):
            state.extend([
                self._blocks[i].position[0], 
                self._blocks[i].position[1],
                self._blocks[i].linearVelocity[0],
                self._blocks[i].linearVelocity[1]
                ])
        return np.array(state).reshape(self.n*self.N_O, 1)

    def step(self, action):
        done = False
        self._current_timestep += 1

        self._gripper.ApplyForceToCenter((float(action[0,0])*self.force_mag, float(action[1,0])*self.force_mag), wake=True)
        self.world.Step(self._dt, 6*30, 2*30)

        # Grasp
        for i in range(self._num_blocks):
            if self._grasps[i] and not self._prev_grasps[i]:       
                self._graspJoints[i] = self.world.CreateRevoluteJoint(
                    bodyA=self._gripper,
                    bodyB=self._blocks[i],
                    anchor=self._gripper.position,
                    lowerAngle = -0.03, # -90 degrees
                    upperAngle = 0.03, #  45 degrees
                    enableLimit = True,
                    maxMotorTorque = 10.0,
                    motorSpeed = 0.0,
                    enableMotor = False,
                    )
        self._prev_grasps = self._grasps.copy()

        done = True if self._current_timestep > self.episode_len else False
        reward = 0
        if self._cfg['scene']['gui']:
            self.render()
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        # - human: render to the current display or terminal and
        #   return nothing. Usually for human consumption.
        # - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
        #   representing RGB values for an x-by-y pixel image, suitable
        #   for turning into a video.
        # - ansi: Return a string (str) or StringIO.StringIO containing a
        #   terminal-style text representation. The text can include newlines
        #   and ANSI escape sequences (e.g. for colors).
        from gym.envs.classic_control import rendering
        ppm = self._cfg['scene']['pixels_per_meter']
        sw = self._cfg['scene']['screen_width']
        sh = self._cfg['scene']['screen_height']
        if self.viewer is None:
            self.viewer = rendering.Viewer(sw, sh)
        self.viewer.set_bounds(-sw/ppm/2, sw/ppm/2, -sh/ppm/2, sh/ppm/2)
        
        # Plotting dynamic objects
        for obj in self._drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # plotting the goal fixture
        t = rendering.Transform(translation=(self.goal_pos[0], self.goal_pos[1]))
        self.viewer.draw_circle(0.1, 30, color=(0, 0.0, 0.0), filled=False, linewidth=3).add_attr(t)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def initial_traj_interpolate(self, x0, xf, T):
        tau = []
        x0 = x0.flatten()
        _xf = x0.copy()
        for i in range(self.N_O-1):
            _xf1 = np.tile(x0.flatten()[(i+1)*self.n:(i+2)*self.n], self.N_O)
            tau.append(np.linspace(_xf, _xf1, T//self.N_O))
            _xf = _xf1.copy()
        
        tau.append(np.linspace(_xf, xf, T - (self.N_O-1)*(T//self.N_O)))
        tau = np.concatenate(tau, axis=0)

        tau = np.append(tau, np.zeros((T, self.m)), axis=1)[:,:,None]
        return tau

    def intialize_cost_matrices(self, zf, n, pick_up_seq):
        Qgoal = 1e-6*np.eye(n*self.N_O) # minimize distance to goal
        Qpick = 1e-6*np.eye(n*self.N_O) # minimize distance between gripper and picked objects
        block_cost = np.diag([3e+0,3e+0,2e-1,2e-1,1e-6])

        # Pick up first object
        Qpick[:n,:n] += 1.0*block_cost
        Qpick[(pick_up_seq[0]+1)*n:(pick_up_seq[0]+2)*n,:n] += -1.0*block_cost
        Qpick[:n,(pick_up_seq[0]+1)*n:(pick_up_seq[0]+2)*n] += -1.0*block_cost
        Qpick[(pick_up_seq[0]+1)*n:(pick_up_seq[0]+2)*n,(pick_up_seq[0]+1)*n:(pick_up_seq[0]+2)*n] += 1.0*block_cost

        # Pick up subsequent object
        for i in range(len(pick_up_seq)-1):
            indx = pick_up_seq[i]
            indx_nxt = pick_up_seq[i+1]

            Qpick[(indx+1)*n:(indx+2)*n,(indx+1)*n:(indx+2)*n] += 1.0*block_cost
            Qpick[(indx+1)*n:(indx+2)*n,(indx_nxt+1)*n:(indx_nxt+2)*n] += -1.0*block_cost
            Qpick[(indx_nxt+1)*n:(indx_nxt+2)*n,(indx+1)*n:(indx+2)*n] += -1.0*block_cost
            Qpick[(indx_nxt+1)*n:(indx_nxt+2)*n,(indx_nxt+1)*n:(indx_nxt+2)*n] += 1.0*block_cost
        
        Qgoal[(indx_nxt+1)*n:(indx_nxt+2)*n, (indx_nxt+1)*n:(indx_nxt+2)*n] += np.diag([5e+1,5e+1,1e-0,1e-0,1e-6]) # Take last block to goal

        R = 1e-2*np.eye(self.m)
        QR = np.zeros((n*self.N_O, self.m))
        RQ = np.zeros((self.m, n*self.N_O))

        cx = -Qgoal@zf
        cu = np.zeros((self.m, 1))
        
        C = np.append( np.append(Qpick+Qgoal, QR, axis=1), np.append(RQ, R, axis=1), axis=0)
        c = np.append(cx, cu, axis=0)

        self.C = np.repeat(C[np.newaxis, :, :], self.episode_len, axis=0)
        self.c = np.repeat(c[np.newaxis, :, :], self.episode_len, axis=0)

    def quadratize_cost(self, tau):
        return self.C, self.c

    def process_input_and_final_state(self, x0, xf):
        return x0.reshape(self.n*self.N_O,1), xf.reshape(self.n*self.N_O,1), x0.reshape(self.n*self.N_O,1), xf.reshape(self.n*self.N_O,1)