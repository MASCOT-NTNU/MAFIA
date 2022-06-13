"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

#% Step I: create wgs coordinates

import pandas as pd
from PreConfig.WaypointGraph.HexagonalWaypoint3D import HexgonalGrid3DGenerator
from Config.Config import *
from PreConfig.WaypointGraph.waypointGraphSetup import DISTANCE_NEIGHBOUR
from usr_func import latlon2xy

polygon_border = FILEPATH + "PreConfig/polygon_border.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = np.empty([10, 2])

depth = [0.5, 1., 1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5, 5]
gridGenerator = HexgonalGrid3DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                                        depth=depth, neighbour_distance=DISTANCE_NEIGHBOUR)
coordinates = gridGenerator.coordinates

#% Step II: convert wgs to xyz

x, y = latlon2xy(coordinates[:, 0], coordinates[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
z = coordinates[:, 2]
xyz = np.vstack((x, y, z)).T
lat_origin = LATITUDE_ORIGIN * np.ones_like(x)
lon_origin = LONGITUDE_ORIGIN * np.ones_like(x)
origin = np.vstack((lat_origin, lon_origin)).T

# dataset = np.hstack((xyz, coordinates, origin))
# df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'lat', 'lon', 'depth', 'lat_origin', 'lon_origin'])
dataset = xyz
df = pd.DataFrame(dataset, columns=['x', 'y', 'z'])
df.to_csv(FILEPATH + "Config/WaypointGraph.csv", index=False)


#%% Step III: check with plot
import plotly.graph_objects as go
import numpy as np
import plotly

# Helix equation
t = np.linspace(0, 20, 100)
x = df['y'].to_numpy()
y = df['x'].to_numpy()
z = df['z'].to_numpy()

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename=FILEPATH+"fig/grid.html", auto_open=True)

#%% Step IV: check with GIS to make sure the grid is produced in the right spot
#TODO: check with GIS after grid discrestisation

#%%
# import matplotlib.pyplot as plt
# plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
# plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-')
# box = np.array([[grid.box_lon_min, grid.box_lat_min],
#                 [grid.box_lon_max, grid.box_lat_min],
#                 [grid.box_lon_max, grid.box_lat_max],
#                 [grid.box_lon_min, grid.box_lat_max]])
# x = grid.grid_x
# y = grid.grid_y
# lat = np.load('MAFIA/models/lats.npy')
# lon = np.load('MAFIA/models/lons.npy')
# depth = np.load('MAFIA/models/debth.npy')

# plt.plot(grid.grid_wgs[:, 1], grid.grid_wgs[:, 0], 'y.')
# plt.plot(box[:, 0], box[:, 1], 'b.')
# plt.plot(lon, lat, 'g*')
# plt.show()


