import pandas as pd

# from GOOGLE.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from GOOGLE.Simulation_2DNidelva.Field.Grid.HexagonalGrid2D import HexgonalGrid2DGenerator
from usr_func import *

PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/Config/OpArea.csv"
PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/Config/Munkholmen.csv"
DISTANCE_LATERAL = 150

polygon = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
munkholmen = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()
gridGenerator = HexgonalGrid2DGenerator(polygon_border=polygon, polygon_obstacle=munkholmen, distance_neighbour=500)
# gridGenerator = GridGenerator(polygon=polygon, depth=[0], distance_neighbour=DISTANCE_LATERAL, no_children=6, points_allowed=5000)
# grid = gridGenerator.grid
coordinates = gridGenerator.coordinates2d

plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(polygon[:, 1], polygon[:, 0], 'r-.')
plt.plot(munkholmen[:, 1], munkholmen[:, 0], 'r-.')
plt.show()


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

