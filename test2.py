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

import scipy.stats
import matplotlib.pyplot as plt

# Simulation window parameters
xMin = 0
xMax = 1
yMin = 0
yMax = 1
xDelta = xMax - xMin
yDelta = yMax - yMin  # rectangle dimensions
areaTotal = xDelta * yDelta

# Point process parameters
lambda0 = 100  # intensity (ie mean density) of the Poisson process

# Simulate Poisson point process
numbPoints = scipy.stats.poisson(lambda0 * areaTotal).rvs()  # Poisson number of points
xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((numbPoints, 1))) + xMin  # x coordinates of Poisson points
yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((numbPoints, 1))) + yMin  # y coordinates of Poisson points
# Plotting
plt.scatter(xx, yy, edgecolor='b', facecolor='none', alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")

plt.show()