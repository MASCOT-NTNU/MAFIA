"""
This script produces GMRF grid
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""
import pandas as pd
from MAFIA.Simulation.Config.Config import *

lats = np.load(FILEPATH + "models/lats.npy")
lons = np.load(FILEPATH + "models/lons.npy")
depth = np.load(FILEPATH + "models/depth.npy")
x, y = latlon2xy(lats, lons, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
z = depth
GMRFGrid = np.vstack((x, y, z)).T

df = pd.DataFrame(GMRFGrid, columns=['x', 'y', 'z'])
df.to_csv(FILEPATH+"Simulation/Config/GMRFGrid.csv", index=False)
print("GMRF grid is generated successfully!")

