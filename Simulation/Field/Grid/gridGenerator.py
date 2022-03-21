"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import numpy as np
import pandas as pd
# from MAFIA.Simulation.Field.Grid.HexagonalGrid2D import HexgonalGrid2DGenerator
from MAFIA.Simulation.Field.Grid.HexagonalGrid3D import HexgonalGrid3DGenerator
from MAFIA.Simulation.Config.Config import *
FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/"

polygon_border = FILEPATH + "Simulation/Config/polygon_border.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = np.empty([10, 2])

depth = [0.5, 1., 1.5, 2., 2.5]
gridGenerator = HexgonalGrid3DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                                        depth=depth, neighbour_distance=DISTANCE_NEIGHBOUR)
coordinates = gridGenerator.coordinates
df = pd.DataFrame(coordinates, columns=['lat', 'lon', 'depth'])
df.to_csv(FILEPATH + "Simulation/Field/Grid/Grid.csv", index=False)
coordinates.shape

import plotly.graph_objects as go
import numpy as np
import plotly

# Helix equation
t = np.linspace(0, 20, 100)
x = coordinates[:, 1]
y = coordinates[:, 0]
z = coordinates[:, 2]

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/fig/grid.html", auto_open=True)


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


