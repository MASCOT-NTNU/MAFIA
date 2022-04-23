
"""
This script tests grid generation
"""
import os
import pandas as pd

working_directory = os.getcwd()

filepath = working_directory + "/MAFIA/"
grid = pd.read_csv(filepath + "Simulation/PreConfig/Grid/Grid.csv").to_numpy()

import numpy as np

from MAFIA.Simulation.PreConfig.Grid.Location import Location
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

#%% plot the corresponding layer with coordinates
import numpy as np
import pandas as pd

path_mafia = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/models/"
lon = np.load(path_mafia+"lons.npy")
lat = np.load(path_mafia+"lats.npy")
depth = np.load(path_mafia+"depth.npy")
grid_box = np.load(path_mafia+"grid.npy")

grid = pd.read_csv("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/Simulation/Field/Grid/Grid.csv").to_numpy()
prior = np.load(path_mafia+"prior.npy")


import plotly.graph_objects as go
import numpy as np
import plotly

# # Helix equation
# t = np.linspace(0, 20, 100)
# x, y, z = np.cos(t), np.sin(t), t

fig = go.Figure(data=[go.Scatter3d(
    x=lon,
    y=lat,
    z=-depth,
    mode='markers',
    marker=dict(
        size=12,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

fig.add_trace(go.Scatter3d(
    x=grid[:, 1],
    y=grid[:, 0],
    z=-grid[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
))

fig.add_trace(go.Scatter3d(
    x=grid_box[:, 3],
    y=grid_box[:, 2],
    z=np.ones_like(grid_box[:, 2]),
    mode='markers',
    marker=dict(
        size=20,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
))

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

plotly.offline.plot(fig, filename='/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/fig/martin.html',
                    auto_open=True)

#%%
import matplotlib.pyplot as plt
plt.plot(grid[:2000, 1], grid[:2000, 0], 'k.')
plt.show()

