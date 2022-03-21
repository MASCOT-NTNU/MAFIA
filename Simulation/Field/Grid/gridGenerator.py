"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import numpy as np
import pandas as pd
from MAFIA.Simulation.Field.Grid.HexagonalGrid2D import HexgonalGrid2DGenerator
from MAFIA.Simulation.Config.Config import *
FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/"

polygon_border = FILEPATH + "Simulation/Config/polygon_border.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = np.empty([10, 2])

grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR)
coordinates = grid.coordinates2d
# coordinates = np.hstack((coordinates, np.zeros_like(coordinates[:, 0].reshape(-1, 1))))
# df = pd.DataFrame(coordinates, columns=['lat', 'lon', 'depth'])
# df.to_csv(FILEPATH + "Field/Grid/Grid.csv", index=False)
# coordinates.shape
import matplotlib.pyplot as plt
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-')
box = np.array([[grid.box_lon_min, grid.box_lat_min],
                [grid.box_lon_max, grid.box_lat_min],
                [grid.box_lon_max, grid.box_lat_max],
                [grid.box_lon_min, grid.box_lat_max]])
x = grid.grid_x
y = grid.grid_y
# lat = np.load('MAFIA/models/lats.npy')
# lon = np.load('MAFIA/models/lons.npy')
depth = np.load('MAFIA/models/debth.npy')

# plt.plot(grid.grid_wgs[:, 1], grid.grid_wgs[:, 0], 'y.')
plt.plot(box[:, 0], box[:, 1], 'b.')
# plt.plot(lon, lat, 'g*')
plt.show()

# coordinates.shape

#%%
plt.figure()
plt.subplot(121)
plt.plot(lon, lat, 'g*')
plt.subplot(122)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.show()



