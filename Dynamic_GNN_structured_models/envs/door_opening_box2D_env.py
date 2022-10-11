import gym
from gym import error, spaces, utils
from gym.utils import colorize, seeding, EzPickle
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, circleShape, revoluteJointDef, contactListener)
import numpy as np
from .utils import *

FPS = 20 #60

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        for i in range(self.env._num_doors):
            if (self.env._gripper==contact.fixtureA.body and self.env._doors[i]==contact.fixtureB.body) or (self.env._doors[i]==contact.fixtureA.body and self.env._gripper==contact.fixtureB.body):
                self.env._grasps[i] = True
    def EndContact(self, contact):
        pass

class DoorOpening(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    def __init__(self, cfg):
        EzPickle.__init__(self)
        self._cfg = cfg
        self.name = 'DoorOpening'
        self.seed()
        self.viewer = None

        self._dt = cfg['scene']['dt']
        self._fps = 1.0 // self._dt

        self.world = Box2D.b2World()
        self.world.gravity = (cfg['scene']['gravity'][0], cfg['scene']['gravity'][1])
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround

        self._num_doors = len(cfg['env_props']['initial_poses']['doors']['positions'])
        self.N_O = 1+self._num_doors
        self.n = 4 + self._num_doors*6
        self.m = 2 # control len
        self.episode_len = cfg['gym']['episode_len']
        self._collision_thresh = 1e-1

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
                            shape=polygonShape(box=(self._width_G, self._width_G)),
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


        self._width_door = cfg['env_props']['dynamics']['doors']['dimx']
        self._height_door = cfg['env_props']['dynamics']['doors']['dimy']
        density_door = cfg['env_props']['dynamics']['doors']['mass']/(self._width_door*self._height_door)
        
        BOX_FD_DOOR = fixtureDef(
                        shape=polygonShape(box=(self._width_door/2, self._height_door/2)),
                        density=density_door,
                        friction=cfg['env_props']['dynamics']['doors']['friction'],
                        restitution=cfg['env_props']['dynamics']['doors']['restitution'])
        door_positions = cfg['env_props']['initial_poses']['doors']['positions']
        self._doors = [self.world.CreateDynamicBody(
                            position = (position[0], 
                                        position[1]),
                            angle = (0),
                            fixtures = BOX_FD_DOOR,
                            fixedRotation = False
                            ) for position in door_positions]
        for door in self._doors:
            door.color1 = (1.0, 0.644, 0.0)
            door.color2 = (0,0,0)
        self._doors[0].color1 = (0.644, 1.0, 0.0)

        self._hinges = [self.world.CreateStaticBody(
                        shapes=circleShape(radius=self._width_door/2),
                        position=(position[0], position[1]+self._height_door/2)
                        ) for position in door_positions]

        self._joint_hinge_doors = [self.world.CreateRevoluteJoint(
                            bodyA=self._doors[i],
                            bodyB=self._hinges[i],
                            anchor=self._hinges[i].position,
                            enableMotor = False,
                            maxMotorTorque = 500.0,
                            collideConnected = False
                            ) for i in range(self._num_doors)]

        self._graspJoints = [None]*self._num_doors

        self._drawlist = [self._gripper] + self._doors

        high = 10
        self.action_space = spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-1.5, -1.5, -high, -high, -high, -high, -high, -high,-high, -high, -high, -high,-2]), np.array([1.5, 1.5, high, high, high, high, high, high, high, high, high, high,2]), dtype=np.float32)

        self.goal_angle = cfg['gym']['goal']['position']
        self.goal_randomize = cfg['gym']['goal']['randomize']
        self.goal_pos_ranges = cfg['gym']['goal']['position_ranges']
        self.goal_pos = None

        self.reset()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self._gripper: return
        self.world.DestroyBody(self._gripper)
        self._gripper = None
        for i in range(self._num_doors):
            self.world.DestroyBody(self._doors[i])
            self.world.DestroyJoint(self._graspJoints[i])
            self.world.DestroyJoint(self._joint_hinge_doors[i])
        self._doors = None

    def reset_to_state(self, gripper_state, doors_state):
        for i in range(self._num_doors):
            if self._graspJoints[i] is not None:
                self.world.DestroyJoint(self._graspJoints[i])
            self.world.DestroyJoint(self._joint_hinge_doors[i])

        self._gripper.position = (gripper_state[0], gripper_state[1])
        self._gripper.linearVelocity = (gripper_state[2], gripper_state[3])
        for i in range(self._num_doors):
            self._doors[i].position = (doors_state[i,0], doors_state[i,1])
            self._doors[i].linearVelocity = (doors_state[i,2], doors_state[i,3])
            self._doors[i].angle = 0.
            self._hinges[i].position = (doors_state[i,0], doors_state[i,1]+self._height_door/2)
            self._hinges[i].linearVelocity = (doors_state[i,2], doors_state[i,3])
        
        self._joint_hinge_doors = [self.world.CreateRevoluteJoint(
                                    bodyA=self._doors[i],
                                    bodyB=self._hinges[i],
                                    anchor=self._hinges[i].position,
                                    enableMotor = False,
                                    maxMotorTorque = 500.0,
                                    collideConnected = False
                                    ) for i in range(self._num_doors)]

        self._prev_grasps = [False]*self._num_doors
        self._grasps = [False]*self._num_doors
        self._graspJoints = [None]*self._num_doors

    def reset(self):
        self._current_timestep = 0

        potential_gripper_pos = np.array(self._cfg['env_props']['initial_poses']['gripper']['position'])
        gripper_pos = potential_gripper_pos.copy()
        gripper_vel = np.array(self._cfg['env_props']['initial_poses']['gripper']['velocity'])
        
        potential_door_pos = [np.array(self._cfg['env_props']['initial_poses']['doors']['positions'][i])
                            for i in range(self._num_doors)]
        door_pos = potential_door_pos.copy()
        door_vel = [np.array(self._cfg['env_props']['initial_poses']['doors']['velocities'][i])
                            for i in range(self._num_doors)]
        
        randomize = self._cfg['env_props']['initial_poses']['gripper']['randomize'] or self._cfg['env_props']['initial_poses']['doors']['randomize']
        if randomize:
            for sample in range(self._cfg['env_props']['initial_poses']['max_samples']):
                if self._cfg['env_props']['initial_poses']['gripper']['randomize']:
                    pose_ranges = self._cfg['env_props']['initial_poses']['gripper']['position_ranges']
                    potential_gripper_pos = np.random.uniform(low=pose_ranges['low'], high=pose_ranges['high'])
                    velocity_ranges = self._cfg['env_props']['initial_poses']['gripper']['velocity_ranges']
                    gripper_vel = np.random.uniform(low=velocity_ranges['low'], high=velocity_ranges['high'])
                if self._cfg['env_props']['initial_poses']['doors']['randomize']:
                    pose_ranges = self._cfg['env_props']['initial_poses']['doors']['position_ranges']
                    velocity_ranges = self._cfg['env_props']['initial_poses']['doors']['velocity_ranges']
                    potential_door_pos = [np.random.uniform(low=pose_ranges['low'], high=pose_ranges['high'])
                                            for _ in range(self._num_doors)]
                    door_vel = [np.random.uniform(low=velocity_ranges['low'], high=velocity_ranges['high'])
                                        for _ in range(self._num_doors)]
                if not self._is_in_collision(potential_gripper_pos, potential_door_pos):
                    gripper_pos = potential_gripper_pos.copy()
                    door_pos = potential_door_pos.copy()
                    break 

        self.reset_to_state(np.append(gripper_pos, gripper_vel), np.append(np.stack(door_pos,axis=0), np.stack(door_vel,axis=0), axis=1))
        
        return self._get_state().squeeze()

    def _is_in_collision(self, gripper_pos, door_poses):
        if self._cfg['env_props']['dynamics']['gripper']['shape'] == 'circle':
            dim_G = self._radius_G + 0.
        if self._cfg['env_props']['dynamics']['gripper']['shape'] == 'box':
            dim_G = self._width_G*np.sqrt(2)

        for door_pos in door_poses:
            box = np.array(
                [[door_pos[0]-dim_G-self._width_door/2-self._collision_thresh, door_pos[0]+dim_G+self._width_door/2+self._collision_thresh],
                [door_pos[1]-dim_G-self._height_door/2-self._collision_thresh, door_pos[1]+dim_G+self._height_door/2+self._collision_thresh]])
            if point_is_in_box(gripper_pos, box):
                return True
        return False

    def _get_state(self):
        state = [
            self._gripper.position[0],
            self._gripper.position[1],
            self._gripper.linearVelocity[0],
            self._gripper.linearVelocity[1]
            ]
        for i in range(self._num_doors):
            state.extend([
                self._doors[i].position[0], 
                self._doors[i].position[1],
                self._doors[i].linearVelocity[0],
                self._doors[i].linearVelocity[1],
                self._hinges[i].position[0],
                self._hinges[i].position[1]
                ])
        return np.array(state)

    def step(self, action):
        done = False
        self._current_timestep += 1

        self._gripper.ApplyForceToCenter((float(action[0]), float(action[1])), wake=True)
        self.world.Step(self._dt, 6*30, 2*30)

        # Grasp
        for i in range(self._num_doors):
            if self._grasps[i] and not self._prev_grasps[i]:
                self._graspJoints[i] = self.world.CreateDistanceJoint(bodyA=self._gripper,
                        bodyB=self._doors[i],
                        anchorA=self._gripper.position,
                        anchorB=self._doors[i].position,
                        collideConnected=True)
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
        self.viewer.draw_circle(0.04, 30, color=(1, 0.0, 0.0), filled=False, linewidth=3).add_attr(t)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def initial_traj_interpolate(self, x0, xf, T):
        self.indx_block_to_pick=0
        xf1 = np.tile(x0.flatten()[(self.indx_block_to_pick+1)*4:(self.indx_block_to_pick+2)*4], self.N_O)

        # xf1 = xf/2
        tau = np.append(np.linspace(x0.flatten(), xf1, int(T/2)),
                        np.linspace(xf1, xf.flatten(), T-int(T/2)), axis=0)
        tau = np.append(tau, np.zeros((T, self.m)), axis=1)[:,:,None]
        return tau

    def intialize_cost_matrices(self, zf, n, door_seq):
        # distance between two objects
        self.indx_block_to_pick = door_seq
        Q1 = np.tile(1e-6*np.eye(n), (self.N_O, self.N_O))
        block_cost = np.diag([6.5e+0,6.5e+0,2e-1,2e-1])
        
        Q1[:n,:n] = 1.0*block_cost
        Q1[(self.indx_block_to_pick+1)*n:(self.indx_block_to_pick+2)*n,:n] = -1.0*block_cost
        Q1[:n,(self.indx_block_to_pick+1)*n:(self.indx_block_to_pick+2)*n] = -1.0*block_cost
        Q1[(self.indx_block_to_pick+1)*n:(self.indx_block_to_pick+2)*n,(self.indx_block_to_pick+1)*n:(self.indx_block_to_pick+2)*n] = 1.0*block_cost

        Q2 = 1e-6*np.eye(n*self.N_O) # distance to goal
        block_cost2 = np.diag([1e+0,1e+0,2e-0,2e-0])
        
        Q2[(self.indx_block_to_pick+1)*n:(self.indx_block_to_pick+2)*n,(self.indx_block_to_pick+1)*n:(self.indx_block_to_pick+2)*n] = 1.0*block_cost2
        
        R = 1e-2*np.eye(self.m)
        QR = np.zeros((n*self.N_O, self.m))
        RQ = np.zeros((self.m, n*self.N_O))

        Q = Q1 + Q2
        C = np.append( np.append(Q, QR, axis=1), np.append(RQ, R, axis=1), axis=0)
        
        cx1 = np.zeros((n*self.N_O,1))
        cx2 = -Q2@zf
        
        cx = cx1 + cx2
        cu = np.zeros((self.m, 1))
        c = np.append(cx, cu, axis=0)

        self.C = np.repeat(C[np.newaxis, :, :], self.episode_len, axis=0)
        self.c = np.repeat(c[np.newaxis, :, :], self.episode_len, axis=0)
        const = zf.reshape(1, n*self.N_O)@(Q1+Q2)@zf.reshape(n*self.N_O, 1)
        self.const = np.repeat(const[np.newaxis, :], self.episode_len, axis=0)

    def quadratize_cost(self, tau):
        return self.C, self.c
    
    def extract_cartesian_state(self, state):
        state_cart = state[:4].copy()
        for i in range(0, self._num_doors):
            state_cart = np.append(state_cart, state[4+6*i:4+6*i+4])
        return state_cart
    
    def extract_full_state(self, state):
        state_cart = state[:4].copy()
        for i in range(0, self._num_doors):
            state_cart = np.append(state_cart, 
                        np.append(state[4+4*i:4+4*i+4],
                        np.array([self._hinges[0].position[0], self._hinges[0].position[1]])
                        ))
        return state_cart