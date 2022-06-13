import os

import numpy as np
import pandas as pd
import plotly
from usr_func import *

import matplotlib.pyplot as plt
FILEPATH = os.getcwd() + "/MAFIA/"
threshold = np.load(FILEPATH + "threshold.npy")
grid = pd.read_csv(FILEPATH + "Simulation/Config/GMRFGrid.csv").to_numpy()
mu2cond = np.load(FILEPATH + "mu2cond.npy")
mucond = np.load(FILEPATH + "mucond.npy")


#%%

fig = go.Figure(data=go.Scatter3d(
    x = grid[:, 1],
    y = grid[:, 0],
    z = -grid[:, 2],
    mode="markers",
    marker=dict(color=mucond, cmin=0, cmax=30)
# marker=dict(color=mu2cond[:-2], cmin=0, cmax=30)
))

plotly.offline.plot(fig, filename=FILEPATH+"check.html", auto_open=True)

