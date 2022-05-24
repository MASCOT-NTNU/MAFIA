"""
This script does simple EDA analysis
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-13
"""
import matplotlib.pyplot as plt

from usr_func import *
from MAFIA.EDA.Config.Config import *
from DataHandler.SINMOD import SINMOD
from skgstat import Variogram


DATAPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/MAFIA/"
SINMODPATH = "/Users/yaolin/HomeOffice/MAFIA/Experiments/20220511/"
FIGPATH = "/Users/yaolin/HomeOffice/MAFIA/fig/Experiments/20220511/"
# AUV_TIMESTEP = 170
# == Set up
LAT_START = 63.456232
LON_START = 10.435198


class EDA:

    def __init__(self):
        self.load_auv_data()
        # self.load_grfar_model()
        # self.load_rrtstar()
        # self.load_cost_valley()

    def load_auv_data(self):
        self.data_auv = pd.read_csv(DATAPATH + "data_sync.csv").to_numpy()
        self.timestamp_auv = self.data_auv[:, 0]
        self.lat_auv = self.data_auv[:, 1]
        self.lon_auv = self.data_auv[:, 2]
        self.depth_auv = self.data_auv[:, 3]
        self.salinity_auv = self.data_auv[:, 4]
        self.temperature_auv = self.data_auv[:, 5]
        print("AUV data is loaded successfully!")

    def load_sinmod_data(self, data_exists=True):
        if not data_exists:
            self.sinmod = SINMOD()
            self.sinmod.load_sinmod_data(raw_data=True)
            coordinates_auv = np.vstack((self.lat_auv, self.lon_auv, self.depth_auv)).T
            self.sinmod.get_data_at_coordinates(coordinates_auv)
        else:
            self.data_sinmod = pd.read_csv(SINMODPATH+"data_sinmod.csv")
            print("SINMOD data is loaded successfully!")
            print(self.data_sinmod.head())
            self.data_sinmod = self.data_sinmod.to_numpy()
        pass

    def plot_scatter_data(self):
        fig = go.Figure(data=go.Scatter3d(
            x=self.lon_auv,
            y=self.lat_auv,
            z=-self.depth_auv,
            mode='markers',
            marker=dict(color=self.data_auv[:, 3], size=10)
        ))
        plotly.offline.plot(fig, filename=FILEPATH+"fig/EDA/samples.html", auto_open=True)
        pass

    def plot_2d(self):
        plt.scatter(self.lon_auv, self.lat_auv, c=self.salinity_auv, cmap=get_cmap("BrBG", 10), vmin=22, vmax=self.CV.threshold)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    e = EDA()
    e.load_sinmod_data(data_exists=False)
    # e.plot_prior()
    # e.plot_sinmod_on_grf_grid()
    # e.plot_sinmod_on_grf_grid()
    # e.plot_2d()
    # e.get_residual_with_sinmod()
    # e.plot_scatter_data()
    # e.plot_recap_mission()
    # e.plot_variogram()

 #%%
# plt.scatter(e.grf_grid[:, 1], e.grf_grid[:, 0], c=e.grfar_model.mu_prior, cmap=get_cmap("BrBG", 10), vmin=10, vmax=27)
# plt.colorbar()
# plt.plot(LON_START, LAT_START, 'rs')
# plt.show()





#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly
import netCDF4
from datetime import datetime
from matplotlib.cm import get_cmap
import re

LATITUDE_ORIGIN = 63.4269097
LONGITUDE_ORIGIN = 10.3969375
CIRCUMFERENCE = 40075000 # [m], circumference
circumference = CIRCUMFERENCE
def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = np.deg2rad((lat - lat_origin)) / 2 / np.pi * CIRCUMFERENCE
    y = np.deg2rad((lon - lon_origin)) / 2 / np.pi * CIRCUMFERENCE * np.cos(np.deg2rad(lat))
    return x, y

file = "/Users/yaolin/Library/CloudStorage/OneDrive-NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2022.05.11.nc"
FIGPATH = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220511/fig/"
# file = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2022.05.10.nc"
# figpath = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220509/fig/"

sinmod = netCDF4.Dataset(file)
ind_before = re.search("samples_", file)
ind_after = re.search(".nc", file)
date_string = file[ind_before.end():ind_after.start()]
ref_timestamp = datetime.strptime(date_string, "%Y.%m.%d").timestamp()
timestamp = np.array(sinmod["time"]) * 24 * 3600 + ref_timestamp #change ref timestamp
lat_sinmod = np.array(sinmod['gridLats'])
lon_sinmod = np.array(sinmod['gridLons'])
depth_sinmod = np.array(sinmod['zc'])
salinity_sinmod = np.array(sinmod['salinity'])

# for i in range(salinity_sinmod.shape[0]):
#     print(i)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(lon_sinmod, lat_sinmod, c=salinity_sinmod[i, 0, :, :], cmap=get_cmap("BrBG", 8), vmin=10, vmax=self.CV.threshold)
#     plt.xlabel("Lon [deg]")
#     plt.ylabel("Lat [deg]")
#     plt.title("SINMOD Surface Salinity Estimation on " + datetime.fromtimestamp(timestamp[i]).strftime("%H:%M, %Y-%m-%d"))
#     plt.colorbar()
#     plt.savefig(figpath+"sinmod/P_{:03d}.jpg".format(i))
#     plt.close("all")

#%%
lat_grid, lon_grid = xy2latlon(e.grf_grid[:, 0], e.grf_grid[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
sal_mean = np.mean(salinity_sinmod[:, 0, :, :], axis=0)
# plt.figure(figsize=(12, 10))
# plt.scatter(lon_sinmod, lat_sinmod, c=sal_mean, cmap=get_cmap("BrBG", 8), vmin=10, vmax=30)
# plt.xlabel("Lon [deg]")
# plt.ylabel("Lat [deg]")
# plt.show()

plt.scatter(lon_grid, lat_grid, c=e.grfar_model.mu_sinmod, cmap=get_cmap("BrBG", 8), vmin=10, vmax=30)
# plt.plot(lon_grid, lat_grid, 'y.', alpha=.3)
plt.colorbar()
plt.title("SINMOD")
# plt.savefig(figpath+"sinmod_grid.jpg")
plt.show()






