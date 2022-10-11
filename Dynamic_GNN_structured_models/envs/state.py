import numpy as np

class State(dict):
    def __init__(self, ball_names, wall_names):
        self._num_objects = len(ball_names) + len(wall_names)
        self.ball_names = ball_names
        self.wall_names = wall_names
        self.object_names = ball_names + wall_names
        self._state_keys = ['pose/position', 
                            'pose/quaternion', 
                            'pose/linear_velocity', 
                            'pose/angular_velocity' , 
                            'constants/mass', 
                            'constants/inertia']
        for object_name in self.object_names:
            # Dynamic features wrt time
            self[f"{object_name}:pose/position"] = None # shape = (3,)
            self[f"{object_name}:pose/quaternion"] = None # shape = (4,)
            self[f"{object_name}:pose/linear_velocity"] = None # shape = (3,)
            self[f"{object_name}:pose/angular_velocity"] = None # shape = (3,)

            # Static features wrt time - shape props
            self[f"{object_name}:constants/friction"] = None # shape = (1,)
            self[f"{object_name}:constants/rolling_friction"] = None # shape = (1,)
            self[f"{object_name}:constants/torsion_friction"] = None # shape = (1,)
            self[f"{object_name}:constants/restitution"] = None # shape = (1,)

            # Static features wrt time - rigid body props
            self[f"{object_name}:constants/mass"] = None # shape = (1,)
            self[f"{object_name}:constants/inertia"] = None # shape = (9,)
        
    def __len__(self):
        return 27*self._num_objects # length of state for each object is 21

    def get_stacked_state(self, state_keys=None):
        if state_keys is None:
            state_keys = self._state_keys
        state_stacked = []
        for object_name in self.object_names:
            for k in state_keys:
                state_stacked.append(self[f"{object_name}:{k}"])
        return np.hstack(state_stacked)
    
    def get_pose_from_state(self, object_name):
        return np.hstack([self[f'{object_name}:pose/position'], self[f'{object_name}:pose/quaternion']])

    def get_vel_from_state(self, object_name):
        return np.hstack([self[f'{object_name}:pose/linear_velocity'], self[f'{object_name}:pose/angular_velocity']])

if __name__ == "__main__":
    state = State(['potato'])
    state['potato:pose/position'] = np.array([1,2,3])
    state['potato:pose/quaternion'] = np.array([1,2,3,4])
    state['potato:pose/linear_velocity'] = np.array([1,2,3])
    state['potato:pose/angular_velocity'] = np.array([1,2,3,4])
    print(state.get_pose_from_state('potato'))
