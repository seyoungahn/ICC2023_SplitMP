import numpy as np
from numpy.random import rand

# Uniform distribution
uniform = lambda MIN, MAX, SAMPLES: rand(*SAMPLES.shape) * (MAX - MIN) + MIN

dimensions = (100, 100)
ndim = len(dimensions)
nodes = np.arange(10)

positions = uniform(np.zeros(ndim), np.array(dimensions), np.dstack((nodes,) * ndim)[0])

print(positions)

class Node(object):
    def __init__(self, id, syst_conf):
        self.id = id
        self.trajectory = []
        self.syst_conf = syst_conf
        self.tx_power = syst_conf['tx_power']
        self.position = None

    def save_trajectory(self, coordi):
        self.trajectory.append(coordi)

    def set_position(self, pos):
        self.position = pos
