from models.mobility import random_waypoint
import numpy as np

"""
- 200 nodes
- Simulation area of 100x100 units
- Velocity chosen from a uniform distribution between 0.1 and 1.0 units/step
- Maximum waiting time of 1.0 steps
"""
# rw = random_waypoint(200, dimensions=(100, 100), velocity=(0.1, 0.1), wt_max=1.0)
#
# positions = next(rw)
# print(np.shape(positions))
# print(positions)