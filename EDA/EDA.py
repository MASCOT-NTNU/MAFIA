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
from MAFIA.EDA.spde import spde
from MAFIA.EDA.Knowledge.Knowledge import Knowledge
from skgstat import Variogram



DATAPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/MAFIA/"
SINMODPATH = "/Users/yaolin/HomeOffice/MAFIA/Experiments/20220511/"
FIGPATH = "/Users/yaolin/HomeOffice/MAFIA/fig/Experiments/20220511/"
# AUV_TIMESTEP = 170
# == Set up
LAT_START = 63.456232
LON_START = 10.435198
IND_END_PRERUN = 1850
AUV_TIMESTEP = 170


@vectorize(['float32(float32, float32, float32)'])
def get_ep(mu, sigma, threshold):
  temp = (threshold - mu)*SQRT1_2 / sigma
  cdf = .5 * (1.+math.erf(temp))
  return cdf


class EDA:

    def __init__(self):
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.update_knowledge()
        self.load_threshold()
        self.load_auv_data()

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
        self.N_gmrf_grid = len(self.gmrf_grid)
        print("E2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2, reduce=True, method=2)
        print("E3: GMRF model is loaded successfully!")

    def update_knowledge(self):
        self.knowledge = Knowledge(gmrf_grid=self.gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
        print("E4: Knowledge of the field is set up successfully!")

    def load_threshold(self):
        self.threshold = np.load(SINMODPATH + "threshold.npy")
        print("E5: threshold is loaded successfully!", self.threshold)

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

    def plot_scatter_data(self):
        fig = go.Figure(data=go.Scatter3d(
            x=self.lon_auv[:IND_END_PRERUN],
            y=self.lat_auv[:IND_END_PRERUN],
            z=-self.depth_auv[:IND_END_PRERUN],
            mode='markers',
            marker=dict(color=self.data_auv[:IND_END_PRERUN, 3], size=10)
        ))
        plotly.offline.plot(fig, filename=FIGPATH+"prerun.html", auto_open=True)
        fig = go.Figure(data=go.Scatter3d(
            x=self.lon_auv[IND_END_PRERUN:],
            y=self.lat_auv[IND_END_PRERUN:],
            z=-self.depth_auv[IND_END_PRERUN:],
            mode='markers',
            marker=dict(color=self.data_auv[IND_END_PRERUN:, 3], size=10)
        ))
        plotly.offline.plot(fig, filename=FIGPATH + "adaptive.html", auto_open=True)

    def plot_sinmod(self):
        fig = go.Figure(data=go.Scatter3d(
            x=self.lon_auv,
            y=self.lat_auv,
            z=-self.depth_auv,
            mode='markers',
            marker=dict(color=self.data_sinmod[:, 3], size=10)
        ))
        plotly.offline.plot(fig, filename=FIGPATH + "sinmod_on_auv_trajectory.html", auto_open=True)
        pass

    def plot_2d(self):
        plt.scatter(self.lon_auv, self.lat_auv, c=self.salinity_auv, cmap=get_cmap("BrBG", 10), vmin=22, vmax=self.CV.threshold)
        plt.colorbar()
        plt.show()

    def plot_prior(self):
        # == plot gmrf section
        xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
        yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
        zrot = -self.gmrf_grid[:, 2]
        ind_plot = np.where((zrot < 0) * (zrot >= -5) * (xrot > 70))[0]

        mu_plot = self.knowledge.mu[ind_plot]
        var_plot = self.knowledge.SigmaDiag[ind_plot]
        ep_plot = get_ep(mu_plot.astype(np.float32), var_plot.astype(np.float32), np.float32(self.gmrf_model.threshold))
        self.yplot = xrot[ind_plot]
        self.xplot = yrot[ind_plot]
        self.zplot = zrot[ind_plot]

        filename = FIGPATH + "Prior.jpg"
        fig_mu = self.plot_figure(mu_plot, filename)
        plotly.offline.plot(fig_mu, filename=FIGPATH + "Prior.html", auto_open=True)

    def plot_recap_mission(self):
        self.trajectory_plot = []
        counter = 0
        xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
        yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
        zrot = -self.gmrf_grid[:, 2]
        ind_plot = np.where((zrot < 0) * (zrot >= -5) * (xrot > 70))[0]

        self.yplot = xrot[ind_plot]
        self.xplot = yrot[ind_plot]
        self.zplot = zrot[ind_plot]

        for i in range(0, len(self.lat_auv), AUV_TIMESTEP):
            counter += 1
            print(counter)
            blockPrint()
            if i+AUV_TIMESTEP <= len(self.lat_auv):
                ind_start = i
                ind_end = i + AUV_TIMESTEP
            else:
                ind_start = i
                ind_end = -1

            x, y = latlon2xy(self.lat_auv[ind_start:ind_end],
                             self.lon_auv[ind_start:ind_end], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
            dataset = np.vstack((x, y,
                                 self.depth_auv[ind_start:ind_end],
                                 self.salinity_auv[ind_start:ind_end])).T
            ind_measured, measurements = self.assimilate_data(dataset)
            for i in range(len(ind_measured)):
                xtemp = (self.gmrf_grid[ind_measured[i], 0] * np.cos(ROTATED_ANGLE) -
                         self.gmrf_grid[ind_measured[i], 1] * np.sin(ROTATED_ANGLE))
                ytemp = (self.gmrf_grid[ind_measured[i], 0] * np.sin(ROTATED_ANGLE) +
                         self.gmrf_grid[ind_measured[i], 1] * np.cos(ROTATED_ANGLE))
                ztemp = -self.gmrf_grid[ind_measured[i], 2]
                self.trajectory_plot.append([ytemp, xtemp, ztemp])

            self.gmrf_model.update(rel=measurements, ks=ind_measured)
            self.update_knowledge()

            enablePrint()
            mu_plot = self.knowledge.mu[ind_plot]
            var_plot = self.knowledge.SigmaDiag[ind_plot]
            ep_plot = get_ep(mu_plot.astype(np.float32), var_plot.astype(np.float32),
                             np.float32(self.threshold))

            filename = FIGPATH + "mu_cond/P_{:03d}.jpg".format(counter)
            fig_mu = self.plot_figure(mu_plot, filename)
            # plotly.offline.plot(fig_mu, filename=FIGPATH + "Prior.html", auto_open=True)
        pass

    def plot_figure(self, value, filename):
        points_int, values_int = interpolate_3d(self.xplot, self.yplot, self.zplot, value)
        fig = make_subplots(rows = 1, cols = 1, specs = [[{'type': 'scene'}]])

        fig.add_trace(go.Volume(
            x=points_int[:, 0],
            y=points_int[:, 1],
            z=points_int[:, 2],
            value=values_int.flatten(),
            isomin=10,
            isomax=30,
            opacity=.4,
            surface_count=10,
            coloraxis="coloraxis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=1
        )

        # fig.add_trace(go.Scatter3d(
        #     x=point_es[:, 0],
        #     y=point_es[:, 1],
        #     z=point_es[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=12,
        #         color=values_es,  # set color to an array/list of desired values
        #         colorscale='Viridis',  # choose a colorscale
        #         opacity=0.8
        #     )
        # ),
        #     row=1, col=1
        # )

        # fig.add_trace(go.Volume(
        #     x=points_es[:, 0],
        #     y=points_es[:, 1],
        #     z=points_es[:, 2],
        #     value=values_es.flatten(),
        #     isomin=0,
        #     isomax=1,
        #     opacity = 0.4,
        #     surface_count = 1,
        #     colorscale = "Reds",
        #     showscale=False,
        #     caps=dict(x_show=False, y_show=False, z_show = False),
        #     ),
        #     row=1, col=1
        # )

        trjp = np.array(self.trajectory_plot)
        fig.add_trace(go.Scatter3d(
            x=trjp[:, 0],
            y=trjp[:, 1],
            z=trjp[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='yellow',
                opacity=.6,
            ),
            # line=dict(
            #     width=1,
            #     color='black',
            # ),
        ),
            row=1, col=1
        )

        fig.update_coloraxes(colorscale="BrBG", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                              tickfont=dict(size=18, family="Times New Roman"),
                                                              title="Salinity",
                                                              titlefont=dict(size=18, family="Times New Roman")),
                             colorbar_x=.95)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=.5)
        )
        # fig.update_layout(coloraxis_colorbar_x=0.8)
        fig.update_layout(
            title={
                'text': "Adaptive 3D myopic illustration",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=30, family="Times New Roman"),
            },
            scene=dict(
                zaxis=dict(nticks=4, range=[-5.5, -0.5], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.4),
            scene_camera=camera,
        )

        # plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=False)
        fig.write_image(filename, width=1980, height=1080)
        return fig

    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[-10:, :])
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= MIN_DEPTH_FOR_DATA_ASSIMILATION)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        print("dataset after filtering: ", dataset[-10:, :])
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 1]).T) ** 2
        dz = ((vectorise(dataset[:, 2]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 2]).T) * GMRF_DISTANCE_NEIGHBOUR) ** 2
        dist = dx + dy + dz
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return ind_assimilated, vectorise(salinity_assimilated)


if __name__ == "__main__":
    e = EDA()
    # e.load_sinmod_data(data_exists=True)
    # e.plot_scatter_data()
    # e.plot_sinmod()
    # e.plot_prior()
    e.plot_recap_mission()
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

#%%
import plotly.graph_objects as go
import numpy as np

# point_es, values_es = interpolate_3d(e.xplot, e.yplot, e.zplot, e.es_plot)
point_es = e.points_es
values_es = e.values_es

fig = go.Figure(data=[go.Scatter3d(
    x=point_es[:, 0],
    y=point_es[:, 1],
    z=point_es[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color=values_es,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename=FIGPATH + "Excursion_set.html", auto_open=True)




