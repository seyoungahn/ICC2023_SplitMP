import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# System configurations
x_min = -50
x_max = 50
y_min = -50
y_max = 50
A = (x_max - x_min) * (y_max - y_min)

# Point process parameters
lambda0 = 0.1  # intensity (i.e. mean density) of the Poisson process

def get_next_waypoint(curr_waypoint):
    ## Selecting next waypoint based on homogeneous PPP
    n_candidates = scipy.stats.poisson(lambda0 * A).rvs()  # Poisson number of points
    xx = scipy.stats.uniform.rvs(x_min, x_max - x_min, (n_candidates, 1))  # x coordinates of Poisson points
    yy = scipy.stats.uniform.rvs(y_min, y_max - y_min, (n_candidates, 1))  # y coordinates of Poisson points
    candidates = np.hstack((xx, yy))
    distances = np.linalg.norm(candidates - curr_waypoint, axis=1)
    return candidates[distances.argmin()]

def get_velocity(v_min, v_max):
    return scipy.stats.uniform.rvs(v_min, v_max - v_min)

def get_pause_time(s_min, s_max):
    return scipy.stats.uniform.rvs(s_min, s_max - s_min)

def get_transition_time(curr_traj):
    # curr_traj = (Xn-1, Xn, Vn, Sn)
    transition_length = np.linalg.norm(curr_traj[1] - curr_traj[0])
    return transition_length / curr_traj[2]