import numpy as np
from isaacgym_utils.math_utils import np_to_transform

def set_pose_vel(pose, vel, name, object_ref, env_idx):
    """
    Throw error if no id exists in this environment with that name
    return transform
    """
    rb_states = object_ref.get_rb_states(env_idx, name)
    for i, k in enumerate('xyz'):
        rb_states['vel']['linear'][k] = vel[i]
        rb_states['vel']['angular'][k] = vel[i+3]

    transforms = [np_to_transform(np.hstack(pose), format="wxyz")]
    for i, transform in enumerate(transforms):
        for k in 'xyz':
            rb_states[i]['pose']['p'][k] = getattr(transform.p, k)

        for k in 'wxyz':
            rb_states[i]['pose']['r'][k] = getattr(transform.r, k)
    object_ref.set_rb_states(env_idx, name, rb_states)

def point_is_in_box(point, box):
    """ Check if a point lies inside a rectangle aligned with the x and y axes (rectangle is not rotated)

    Keyword arguments:
    point -- an array/list of the form [x, y]
    box --  an array of the form [[xmin, xmax], [ymin, ymax]]

    Return:
    boolean -- True or False
    """

    xmin = np.min(box[0,:])
    xmax = np.max(box[0,:])

    ymin = np.min(box[1,:])
    ymax = np.max(box[1,:])

    if point[0] > xmin and point[0] < xmax and point[1] > ymin and point[1] < ymax:
        return True
    else:
        return False

def find_edge_index(n):
    # Ball-wall pairs
    from_vec = np.repeat(np.arange(n), n)
    to_vec = np.tile(np.arange(n), n)
    edge_index = np.stack([from_vec, to_vec], axis=0)
    return edge_index