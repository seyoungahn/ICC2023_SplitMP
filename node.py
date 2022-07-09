import numpy as np
from numpy.random import rand
import scipy
import scipy.stats

# dimensions = (100, 100)
# ndim = len(dimensions)
# nodes = np.arange(10)

# positions = uniform(np.zeros(ndim), np.array(dimensions), np.dstack((nodes,) * ndim)[0])

# print(positions)

class UE(object):
    def __init__(self, id, params):
        self.id = id
        self.trajectory = []
        self.params = params
        self.tx_power = params.syst_tx_power

    def save_trajectory(self, coordi):
        """
        Quadruple (Xn-1, Xn, Vn, Sn):
            Xn-1: the starting waypoint
            Xn: the target waypoint
            Vn: the velocity
            Sn: the pause time at the waypoint Xn
        """

        self.trajectory.append(coordi)

    def set_next_trajectory(self):
        Xn_1 = self.trajectory[-1][1]
        Xn = self.get_next_waypoint(Xn_1)
        Vn = self.get_velocity(self.params.syst_vel[0], self.params.syst_vel[1])
        Sn = self.get_pause_time(self.params.syst_pause[0], self.params.syst_pause[1])
        self.trajectory.append([Xn_1, Xn, Vn, Sn])

    def get_next_waypoint(self, curr_waypoint):
        ## Selecting next waypoint based on homogeneous PPP
        x_min = -self.params.syst_dim[0] / 2
        x_max = self.params.syst_dim[0] / 2
        y_min = -self.params.syst_dim[1] / 2
        y_max = self.params.syst_dim[1] / 2
        A = (x_max - x_min) * (y_max - y_min)
        n_candidates = scipy.stats.poisson(self.params.syst_lambda * A).rvs()  # Poisson number of points
        xx = scipy.stats.uniform.rvs(x_min, x_max - x_min, (n_candidates, 1))  # x coordinates of Poisson points
        yy = scipy.stats.uniform.rvs(y_min, y_max - y_min, (n_candidates, 1))  # y coordinates of Poisson points
        candidates = np.hstack((xx, yy))
        distances = np.linalg.norm(candidates - curr_waypoint, axis=1)
        return candidates[distances.argmin()]

    def get_velocity(self, curr_waypoint, next_waypoint):
        v = scipy.stats.uniform.rvs(self.params.syst_vel[0], self.params.syst_vel[1] - self.params.syst_vel[0])
        direction = (next_waypoint - curr_waypoint) / np.linalg.norm(next_waypoint - curr_waypoint)
        return v * direction

    def get_pause_time(self, s_min, s_max):
        return scipy.stats.uniform.rvs(s_min, s_max - s_min)

    def get_transition_time(self, curr_traj):
        # curr_traj = (Xn-1, Xn, Vn, Sn)
        transition_length = np.linalg.norm(curr_traj[1] - curr_traj[0])
        return transition_length / curr_traj[2]

class AP(object):
    def __init__(self, id, params):
        self.id = id
        self.params = params
        self.position = None