# This module is implemented based on https://github.com/panisson/pymobility
import numpy as np
from numpy.random import rand
from system import System

# define a Uniform Distribution
U = lambda MIN, MAX, SAMPLES: rand(*SAMPLES.shape) * (MAX - MIN) + MIN

# define a Truncated Power Law Distribution
P = lambda ALPHA, MIN, MAX, SAMPLES: ((MAX ** (ALPHA + 1.0) - 1.0) * rand(*SAMPLES.shape) + 1.0) ** (1.0 / (ALPHA + 1.0))

# define an Exponential Distribution
E = lambda SCALE, SAMPLES: -SCALE * np.log(rand(*SAMPLES.shape))

# Palm state probability
def pause_probability_init(pause_low, pause_high, speed_low, speed_high, dimensions):
    alpha1 = ((pause_high + pause_low) * (speed_high - speed_low)) / (2 * np.log(speed_high / speed_low))
    delta1 = np.sqrt(np.sum(np.square(dimensions)))
    return alpha1 / (alpha1 + delta1)


# Palm residual
def residual_time(mean, delta, shape=(1,)):
    t1 = mean - delta
    t2 = mean + delta
    u = rand(*shape)
    residual = np.zeros(shape)
    if delta != 0.0:
        case_1_u = u < (2.0 * t1 / (t1 + t2))
        residual[case_1_u] = u[case_1_u] * (t1 + t2) / 2.0
        residual[np.logical_not(case_1_u)] = t2 - np.sqrt((1. - u[np.logical_not(case_1_u)]) * (t2 * t2 - t1 * t1))
    else:
        residual = u * mean
    return residual


# Initial speed
def initial_speed(speed_mean, speed_delta, shape=(1,)):
    v0 = speed_mean - speed_delta
    v1 = speed_mean + speed_delta
    u = rand(*shape)
    return pow(v1, u) / pow(v0, u - 1)


def init_random_waypoint(nr_nodes, dimensions, speed_low, speed_high, pause_low, pause_high):
    ndim = len(dimensions)
    positions = np.empty((nr_nodes, ndim))
    waypoints = np.empty((nr_nodes, ndim))
    speed = np.empty(nr_nodes)
    pause_time = np.empty(nr_nodes)

    speed_low = float(speed_low)
    speed_high = float(speed_high)

    moving = np.ones(nr_nodes)
    speed_mean, speed_delta = (speed_low + speed_high) / 2.0, (speed_high - speed_low) / 2.0
    pause_mean, pause_delta = (pause_low + pause_high) / 2.0, (pause_high - pause_low) / 2.0

    # steady-state pause probability for Random Waypoint
    q0 = pause_probability_init(pause_low, pause_high, speed_low, speed_high, dimensions)

    for i in range(nr_nodes):

        while True:

            z1 = rand(ndim) * np.array(dimensions)
            z2 = rand(ndim) * np.array(dimensions)

            if rand() < q0:
                moving[i] = 0.0
                break
            else:
                # r is a ratio of the length of the randomly chosen path over
                # the length of a diagonal across the simulation area
                r = np.sqrt(np.sum((z2 - z1) ** 2) / np.sum(np.array(dimensions) ** 2))
                if rand() < r:
                    moving[i] = 1.0
                    break

        positions[i] = z1
        waypoints[i] = z2

    # steady-state positions
    # initially the node has traveled a proportion u2 of the path from (x1,y1) to (x2,y2)
    u2 = rand(*positions.shape)
    positions = u2 * positions + (1 - u2) * waypoints

    # steady-state speed and pause time
    paused_bool = moving == 0.0
    paused_idx = np.where(paused_bool)[0]
    pause_time[paused_idx] = residual_time(pause_mean, pause_delta, paused_idx.shape)
    speed[paused_idx] = 0.0

    moving_bool = np.logical_not(paused_bool)
    moving_idx = np.where(moving_bool)[0]
    pause_time[moving_idx] = 0.0
    speed[moving_idx] = initial_speed(speed_mean, speed_delta, moving_idx.shape)

    return positions, waypoints, speed, pause_time

class MobilityDataGenerator(object):
    def __init__(self, params):
        self.n_nodes = params.n_nodes
        self.syst_dim = params.syst_dim
        self.syst_vel = params.syst_vel
        self.max_wait = params.syst_max_wait
        self.init_stationary = params.syst_init_stationary

    def __iter__(self):
        ndim = len(self.syst_dim)
        min_vel, max_vel = self.syst_vel[0], self.syst_vel[1]

        min_wait_time = 0.0

        if self.init_stationary:
            positions, waypoints, velocity, wait_time = init_random_waypoint(self.n_nodes, self.syst_dim, min_vel, max_vel, min_wait_time, (self.max_wait if self.max_wait is not None else 0.0))
        else:
            nodes = np.arange(self.n_nodes)
            positions = U(np.zeros(ndim), np.array(self.syst_dim), np.dstack((nodes,) * ndim)[0])
            waypoints = U(np.zeros(ndim), np.array(self.syst_dim), np.dstack((nodes,) * ndim)[0])
            wait_time = np.zeros(self.n_nodes)
            velocity = U(min_vel, max_vel, nodes)

        # Assign nodes' movements (direction * velocity)
        direction = waypoints - positions
        direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]

        while True:
            # update node position
            positions += direction * velocity[:, np.newaxis]
            # calculate distance to waypoint
            d = np.sqrt(np.sum(np.square(waypoints - positions), axis=1))
            # update info for arrived nodes
            arrived = np.where(np.logical_and(d<=velocity, wait_time<=0.0))[0]

            # step back for nodes that surpassed waypoint
            positions[arrived] = waypoints[arrived]

            if self.max_wait:
                velocity[arrived] = 0.0
                wait_time[arrived] = U(0, self.max_wait, arrived)
                # update info for paused nodes
                wait_time[np.where(velocity == 0.0)[0]] -= 1.0
                # update info for moving nodes
                arrived = np.where(np.logical_and(velocity == 0.0, wait_time < 0.0))[0]

            if arrived.size > 0:
                waypoints[arrived] = U(np.zeros(ndim), np.array(self.syst_dim), np.zeros((arrived.size, ndim)))
                velocity[arrived] = U(min_vel, max_vel, arrived)

                new_direction = waypoints[arrived] - positions[arrived]
                direction[arrived] = new_direction / np.linalg.norm(new_direction, axis=1)[:, np.newaxis]

            self.velocity = velocity
            self.wt = wait_time
            yield positions

    def set_system(self):
        self.system = System(self.params)
        ## TODO: (1) Distributing APs and CPUs of cell-free massive MIMO systems

        ## TODO: (2) Distributing UEs in the systems

    def __iter__(self):
        while True:
            measurements = {}
            ## TODO: (3-1) System measurements - next waypoint predictions

            ## TODO: (3-2) System measurements - channel predictions

            ## TODO: (3-3) UE movements based on RWP mobility model

            yield measurements