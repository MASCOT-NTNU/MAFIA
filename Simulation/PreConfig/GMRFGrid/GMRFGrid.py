"""
This script produces GMRF grid
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""
import pandas as pd
from MAFIA.Simulation.Config.Config import *

lats = np.load(FILEPATH + "models/lats_small.npy")
lons = np.load(FILEPATH + "models/lons_small.npy")
depth = np.load(FILEPATH + "models/depth_small.npy")
# lats = np.load(FILEPATH + "models/lats.npy")
# lons = np.load(FILEPATH + "models/lons.npy")
# depth = np.load(FILEPATH + "models/depth.npy")
x, y = latlon2xy(lats, lons, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
z = depth
GMRFGrid = np.vstack((x, y, z)).T

df = pd.DataFrame(GMRFGrid, columns=['x', 'y', 'z'])
df.to_csv(FILEPATH+"Simulation/Config/GMRFGrid.csv", index=False)
print("GMRF grid is generated successfully!")


#%% check prior data

import plotly.graph_objects as go
import numpy as np

mu = pd.read_csv(FILEPATH+"Simulation/Config/Data/data_mu_truth.csv")['salinity'].to_numpy()

#%%
# Helix equation
value = mu

fig = go.Figure(data=[go.Scatter3d(
    x=y,
    y=x,
    z=-z,
    mode='markers',
    marker=dict(
        size=12,
        color=value,                # set color to an array/list of desired values
        colorscale='BrBG',   # choose a colorscale
        # opacity=0.8,
        cmin=16,
        cmax=32
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
import plotly
plotly.offline.plot(fig, filename=FILEPATH+"/fig/data.html", auto_open=True)


