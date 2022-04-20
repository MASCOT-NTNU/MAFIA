
"""
This script tests grid generation
"""
import os
import pandas as pd

working_directory = os.getcwd()

filepath = working_directory + "/MAFIA/"
grid = pd.read_csv(filepath + "Simulation/Field/Grid/Grid.csv").to_numpy()

import numpy as np

from MAFIA.Simulation.Field.Grid.Location import Location
loc = Location(324, 234, 0)


def get_ind_at_location3d_xyz(coordinates, location):
    dist_x = coordinates[:, 0] - location.x
    dist_y = coordinates[:, 1] - location.y
    dist_z = coordinates[:, 2] - location.z
    dist = dist_x ** 2 + dist_y ** 2 + dist_z ** 2
    ind = np.where(dist == np.amin(dist))[0]
    return ind
import time

t1 = time.time()
ind = get_ind_at_location3d_xyz(grid[:, -3:], loc)
t2 = time.time()
print("Time consumed: on CPU: ", t2 - t1)


from numba import jit
@jit(nopython=True)
def get_ind_at_location3d_xyz_jit(coordinates, x, y, z):
    dist_x = coordinates[:, 0] - x
    dist_y = coordinates[:, 1] - y
    dist_z = coordinates[:, 2] - z

    dist = dist_x ** 2 + dist_y ** 2 + dist_z ** 2
    print(dist.shape)
    ind = np.where(dist == np.amin(dist))[0]
    return ind


ind = get_ind_at_location3d_xyz_jit(grid[:, -3:], loc.x, loc.y, loc.z)

t1 = time.time()
ind = get_ind_at_location3d_xyz_jit(grid[:, -3:], loc.x, loc.y, loc.z)
t2 = time.time()
print("Time consumed: on para: ", t2 - t1)
print(ind)


