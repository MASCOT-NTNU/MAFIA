import numpy as np
import pandas as pd

from DataHandler.SINMOD import SINMOD
from usr_func import *
FILEPATH = "/Users/yaolin/HomeOffice/MAFIA/EDA/"
FIGPATH = FILEPATH + "fig/"
# FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/"
# FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/EDA/fig/"
# data_auv = pd.read_csv(FILEPATH + "AUVData.csv")
# lat_auv = data_auv['lat'].to_numpy()
# lon_auv = data_auv['lon'].to_numpy()
# depth_auv = data_auv['depth'].to_numpy()
# coordinates_auv = np.vstack((lat_auv, lon_auv, depth_auv)).T
# salinity_auv = data_auv['salinity'].to_numpy()
# df = pd.DataFrame(np.vstack((lat_auv, lon_auv, depth_auv, salinity_auv)).T, columns=['lat', 'lon', 'depth', 'salinity'])
# df.to_csv(FILEPATH + "auv.csv", index=False)

# sinmod = SINMOD()
# filenames_fullpath = ["/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2021.05.27.nc"]
# sinmod.load_sinmod_data(raw_data=True, filenames=filenames_fullpath)
# sinmod.get_data_at_coordinates(coordinates_auv, filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/EDA/sinmod.csv")

data_auv = pd.read_csv(FILEPATH + "auv.csv").to_numpy()
data_sinmod = pd.read_csv(FILEPATH + "sinmod.csv").to_numpy()


#%%

from pathlib import Path
fig = go.Figure(data=[go.Scatter3d(
    x=data_auv[:, 1],
    y=data_auv[:, 0],
    z=-data_auv[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color=data_auv[:, 3],                # set color to an array/list of desired values
        colorscale='BrBG',   # choose a colorscale
        opacity=0.8
    )
)])


# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
figname = FIGPATH+"data_auv.html"
plotly.offline.plot(fig, filename=figname, auto_open=False)
os.system('open -a \"Google Chrome\" '+figname)


#%%
fig = go.Figure(data=[go.Scatter3d(
    x=data_sinmod[:, 1],
    y=data_sinmod[:, 0],
    z=-data_sinmod[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color=data_sinmod[:, 3],                # set color to an array/list of desired values
        colorscale='BrBG',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
figname = FIGPATH+"data_sinmod.html"
plotly.offline.plot(fig, filename=figname, auto_open=False)
os.system('open -a \"Google Chrome\" '+figname)

#%%



