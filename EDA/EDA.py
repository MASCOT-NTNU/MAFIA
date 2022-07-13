"""
This script does simple EDA analysis
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-13
"""
import matplotlib.pyplot as plt
import pandas as pd

from usr_func import *
from MAFIA.EDA.Config.Config import *
from DataHandler.SINMOD import SINMOD
from MAFIA.EDA.spde import spde
from MAFIA.EDA.Knowledge.Knowledge import Knowledge

from skgstat import Variogram



DATAPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/MAFIA/"
SINMODPATH = "/Users/yaolin/HomeOffice/MAFIA/Experiments/20220511/"
FIGPATH = "/Users/yaolin/HomeOffice/MAFIA/fig/Experiments/20220511/"
RAW_SINMODPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2022.05.11.nc"
# AUV_TIMESTEP = 170
# == Set up
LAT_START = 63.456232
LON_START = 10.435198
IND_END_PRERUN = 1850
AUV_TIMESTEP = 170
VMIN = 10
VMAX = 33


@vectorize(['float32(float32, float32, float32)'])
def get_ep(mu, sigma, threshold):
  temp = (threshold - mu)*SQRT1_2 / sigma
  cdf = .5 * (1.+math.erf(temp))
  return cdf


class EDA:

    def __init__(self):
        self.load_polygons()
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.update_knowledge()
        self.load_threshold()
        self.load_waypoint()
        self.load_auv_data()

    def load_polygons(self):
        self.polygon_border = pd.read_csv(FILEPATH + "Config/polygon_border.csv").to_numpy()
        self.polygon_obstacle = pd.read_csv(FILEPATH + "Config/polygon_obstacle.csv").to_numpy()
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)
        print("E1: polygons are loaded successfully!")
        pass

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

    def load_waypoint(self):
        self.waypoint = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        print("E6: Waypoint is loaded successfully!")

    def load_auv_data(self):
        self.data_auv = pd.read_csv(DATAPATH + "data_sync.csv").to_numpy()
        self.timestamp_auv = self.data_auv[:, 0]
        self.lat_auv = self.data_auv[:, 1]
        self.lon_auv = self.data_auv[:, 2]
        self.depth_auv = self.data_auv[:, 3]
        self.salinity_auv = self.data_auv[:, 4]
        self.temperature_auv = self.data_auv[:, 5]
        print("E7: AUV data is loaded successfully!")

    def load_sinmod_data_along_auv_path(self, data_exists=True):
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

    def save_grid_to_gis(self):
        ind_surface = np.where(self.gmrf_grid[:, 2] == .5)[0]
        x = self.gmrf_grid[ind_surface, 0]
        y = self.gmrf_grid[ind_surface, 1]
        lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        gmrf_grid = np.vstack((lat, lon)).T
        df = pd.DataFrame(gmrf_grid, columns=['lat', 'lon'])
        df.to_csv(FILEPATH + "../GIS/csv/gmrf_grid.csv", index=False)

        ind_surface = np.where(self.waypoint[:, 2] == .5)[0]
        x = self.waypoint[ind_surface, 0]
        y = self.waypoint[ind_surface, 1]
        lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        gmrf_grid = np.vstack((lat, lon)).T
        df = pd.DataFrame(gmrf_grid, columns=['lat', 'lon'])
        df.to_csv(FILEPATH + "../GIS/csv/waypoint.csv", index=False)

        prerun = np.vstack((self.lat_auv[:IND_END_PRERUN],
                        self.lon_auv[:IND_END_PRERUN],
                        self.depth_auv[:IND_END_PRERUN],
                        self.salinity_auv[:IND_END_PRERUN])).T
        df = pd.DataFrame(prerun, columns=['lat', 'lon', 'depth', 'salinity'])
        df.to_csv(FILEPATH + "../GIS/csv/auv_prerun.csv", index=False)

        adaptive = np.vstack((self.lat_auv[IND_END_PRERUN:],
                        self.lon_auv[IND_END_PRERUN:],
                        self.depth_auv[IND_END_PRERUN:],
                        self.salinity_auv[IND_END_PRERUN:])).T
        df = pd.DataFrame(adaptive, columns=['lat', 'lon', 'depth', 'salinity'])
        df.to_csv(FILEPATH + "../GIS/csv/auv_adaptive.csv", index=False)
        print("Dataset is saved successfully!")

    def plot_time_series(self):
        tf = np.vectorize(datetime.fromtimestamp)
        tid = tf(self.timestamp_auv)
        # tid_str = []
        # for i in range(len(tid)):
        #     tid_str.append(datetime.strftime(tid[i], "%H:%M"))
        tid_str = tid

        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(nrows=3, ncols=1)
        ax = fig.add_subplot(gs[0])
        ax.plot(tid_str, self.depth_auv, 'k-')
        ax.set_xlim([tid_str[0], tid_str[-1]])
        ax.set_ylabel("Depth [m]")
        ax.set_title("2022-05-11 MAFIA timeseries data")
        ax.grid()

        ax = fig.add_subplot(gs[1])
        ax.plot(tid_str, self.salinity_auv, 'k-')
        ax.set_xlim([tid_str[0], tid_str[-1]])
        ax.set_ylabel("Salinity [psu]")
        ax.grid()

        ax = fig.add_subplot(gs[2])
        ax.plot(tid_str, self.temperature_auv, 'k-')
        ax.set_xlim([tid_str[0], tid_str[-1]])
        ax.set_ylabel("Temperature [deg]")
        ax.set_xlabel("Time")
        ax.grid()
        # rotate and align the tick labels so they look better
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        plt.savefig(FIGPATH + "Timeseries_20220511_MAFIA.pdf")
        plt.show()
        pass
        print("Finished time series plotting")

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

    def plot_sinmod_layer(self):
        file = RAW_SINMODPATH
        ind_before = re.search("samples_", file)
        ind_after = re.search(".nc", file)
        self.date_string = file[ind_before.end():ind_after.start()]
        print(self.date_string)
        ref_timestamp = datetime.strptime(self.date_string, "%Y.%m.%d").timestamp()
        self.sinmod_raw = netCDF4.Dataset(RAW_SINMODPATH)
        self.timestamp = np.array(self.sinmod_raw["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp
        self.lat_sinmod = np.array(self.sinmod_raw['gridLats'])
        self.lon_sinmod = np.array(self.sinmod_raw['gridLons'])
        self.depth_sinmod = np.array(self.sinmod_raw['zc'])
        self.salinity_sinmod = np.array(self.sinmod_raw['salinity'])

        # plot time steps
        # for j in range(self.salinity_sinmod.shape[0]):
        #     print(j)
        #     fig = plt.figure(figsize=(35, 15))
        #     gs = GridSpec(nrows=2, ncols=5)
        #     for i in range(10):
        #         ax = fig.add_subplot(gs[i])
        #         # im = ax.scatter(self.lon_sinmod, self.lat_sinmod, c=self.salinity_sinmod[0, i, :, :], cmap=get_cmap("BrBG", 10),
        #         #            vmin=10, vmax=30)
        #         sal = self.salinity_sinmod[j, i, :, :]
        #         levels = np.arange(VMIN, VMAX)
        #         plt.contour(self.lon_sinmod, self.lat_sinmod, sal, levels=levels, cmap=get_cmap("BrBG", 10), vmin=VMIN, vmax=VMAX)
        #         CS = plt.contour(self.lon_sinmod, self.lat_sinmod, sal, levels=[e.threshold.item()], colors=('r',), linestyles=('-',),
        #                          linewidths=(2,))
        #         plt.contour(self.lon_sinmod, self.lat_sinmod, sal, levels=[0], colors=('w',), linestyles=('-',), linewidths=(2,))
        #         plt.contourf(self.lon_sinmod, self.lat_sinmod, sal, levels=levels, cmap=get_cmap("BrBG", 10), vmin=VMIN, vmax=VMAX)
        #         plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-', linewidth=2)
        #         plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'k-', linewidth=2)
        #         # plt.colorbar()
        #         plt.ylim([np.amin(self.polygon_border[:, 0]), np.amax(self.polygon_border[:, 0])])
        #         ax.set_title("Salinity field at {:.1f}m".format(self.depth_sinmod[i]))
        #         if (i+1) % 5 == 0:
        #             plt.colorbar()
        #         if i >= 5:
        #             ax.set_xlabel("Longitude [deg]")
        #         if i == 0 or i == 5:
        #             ax.set_ylabel("Latitude [deg]")
        #     plt.suptitle("Salinity field on " + datetime.fromtimestamp(e.timestamp[j]).strftime("%Y-%m-%d, %H:%M"))
        #     plt.savefig(FIGPATH + "SINMOD/P_{:03d}.jpg".format(j))
        #     plt.close("all")
        # # plt.show()

        # plot mean
        # for j in range(self.salinity_sinmod.shape[0]):
        fig = plt.figure(figsize=(35, 15))
        gs = GridSpec(nrows=2, ncols=5)
        for i in range(10):
            ax = fig.add_subplot(gs[i])
            # im = ax.scatter(self.lon_sinmod, self.lat_sinmod, c=self.salinity_sinmod[0, i, :, :], cmap=get_cmap("BrBG", 10),
            #            vmin=10, vmax=30)
            sal = np.mean(self.salinity_sinmod[:, i, :, :], axis=0)
            levels = np.arange(VMIN, VMAX)
            plt.contour(self.lon_sinmod, self.lat_sinmod, sal, levels=levels, cmap=get_cmap("BrBG", 10), vmin=VMIN, vmax=VMAX)
            CS = plt.contour(self.lon_sinmod, self.lat_sinmod, sal, levels=[e.threshold.item()], colors=('r',), linestyles=('-',),
                             linewidths=(2,))
            plt.contour(self.lon_sinmod, self.lat_sinmod, sal, levels=[0], colors=('w',), linestyles=('-',), linewidths=(2,))
            plt.contourf(self.lon_sinmod, self.lat_sinmod, sal, levels=levels, cmap=get_cmap("BrBG", 10), vmin=VMIN, vmax=VMAX)
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-', linewidth=2)
            plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'k-', linewidth=2)
            # plt.colorbar()
            plt.ylim([np.amin(self.polygon_border[:, 0]), np.amax(self.polygon_border[:, 0])])
            ax.set_title("Salinity field at {:.1f}m".format(self.depth_sinmod[i]))
            if (i+1) % 5 == 0:
                plt.colorbar()
            if i >= 5:
                ax.set_xlabel("Longitude [deg]")
            if i == 0 or i == 5:
                ax.set_ylabel("Latitude [deg]")
        plt.suptitle("Salinity time-average from SINMOD on 2022-05-11")
        plt.savefig(FIGPATH + "SINMOD_AVE.jpg")
        plt.close("all")
        # plt.show()

        os.system("say finished")
        pass

    def plot_sinmod(self):
        file = RAW_SINMODPATH
        ind_before = re.search("samples_", file)
        ind_after = re.search(".nc", file)
        self.date_string = file[ind_before.end():ind_after.start()]
        print(self.date_string)
        ref_timestamp = datetime.strptime(self.date_string, "%Y.%m.%d").timestamp()
        self.sinmod_raw = netCDF4.Dataset(RAW_SINMODPATH)
        self.timestamp = np.array(self.sinmod_raw["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp
        self.lat_sinmod = np.array(self.sinmod_raw['gridLats'])
        self.lon_sinmod = np.array(self.sinmod_raw['gridLons'])
        self.depth_sinmod = np.array(self.sinmod_raw['zc'])
        self.salinity_sinmod = np.array(self.sinmod_raw['salinity'])

        xsinmod, ysinmod = latlon2xy(self.lat_sinmod, self.lon_sinmod, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        xrot = xsinmod * np.cos(ROTATED_ANGLE) - ysinmod * np.sin(ROTATED_ANGLE)
        yrot = xsinmod * np.sin(ROTATED_ANGLE) + ysinmod * np.cos(ROTATED_ANGLE)
        self.grid_plot = []
        self.sal_plot = []
        self.ind_illegal = np.zeros([xrot.shape[0] * xrot.shape[1] * 10, 1])

        counter = 0
        for i in range(10):
            for j in range(xrot.shape[0]):
                for k in range(xrot.shape[1]):
                    if not self.is_location_legal(xsinmod[j, k], ysinmod[j, k]):
                        self.ind_illegal[counter] = True
                    counter += 1
                    self.grid_plot.append([yrot[j, k], xrot[j, k], -self.depth_sinmod[i]])

        for t in range(self.salinity_sinmod.shape[0]):
            temp = []
            for i in range(10):
                for j in range(xrot.shape[0]):
                    for k in range(xrot.shape[1]):
                        temp.append(self.salinity_sinmod[t, i, j, k])
            self.sal_plot.append([temp])

        # for i in range(self.salinity_sinmod.shape[0]):
        #
        #     points_int, values_int = interpolate_3d(, self.yplot, self.zplot, value)
        #     fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        #
        #     fig.add_trace(go.Volume(
        #         x=points_int[:, 0],
        #         y=points_int[:, 1],
        #         z=points_int[:, 2],
        #         value=values_int.flatten(),
        #         isomin=vmin,
        #         isomax=vmax,
        #         opacity=opacity,
        #         surface_count=surface_count,
        #         coloraxis="coloraxis",
        #         caps=dict(x_show=False, y_show=False, z_show=False),
        #     ),
        #         row=1, col=1
        #     )
        #
        #     try:
        #         trjp = np.array(self.sampling_location_plot)
        #         fig.add_trace(go.Scatter3d(
        #             x=trjp[:, 0],
        #             y=trjp[:, 1],
        #             z=trjp[:, 2],
        #             mode='markers',
        #             marker=dict(
        #                 size=5,
        #                 color='yellow',
        #                 opacity=.6,
        #             ),
        #         ),
        #             row=1, col=1
        #         )
        #     except:
        #         pass
        #
        #     try:
        #         trjp = np.array(self.trajectory_plot)
        #         fig.add_trace(go.Scatter3d(
        #             x=trjp[:, 0],
        #             y=trjp[:, 1],
        #             z=trjp[:, 2],
        #             mode='lines',
        #             line=dict(
        #                 width=1,
        #                 color='black',
        #             ),
        #         ),
        #             row=1, col=1
        #         )
        #     except:
        #         pass
        #
        #     fig.update_coloraxes(colorscale=cmap, colorbar=dict(lenmode='fraction', len=.5, thickness=20,
        #                                                         tickfont=dict(size=18, family="Times New Roman"),
        #                                                         title=cbar_title,
        #                                                         titlefont=dict(size=18, family="Times New Roman")),
        #                          colorbar_x=.95)
        #     camera = dict(
        #         up=dict(x=0, y=0, z=1),
        #         center=dict(x=0, y=0, z=0),
        #         eye=dict(x=-1.25, y=0, z=.5)
        #     )
        #     fig.update_layout(
        #         title={
        #             'text': "Adaptive 3D myopic illustration",
        #             'y': 0.9,
        #             'x': 0.5,
        #             'xanchor': 'center',
        #             'yanchor': 'top',
        #             'font': dict(size=30, family="Times New Roman"),
        #         },
        #         scene=dict(
        #             xaxis=dict(range=[np.amin(points_int[:, 0]), np.amax(points_int[:, 0])]),
        #             yaxis=dict(range=[np.amin(points_int[:, 1]), np.amax(points_int[:, 1])]),
        #             zaxis=dict(nticks=4, range=[-6, .5], ),
        #             xaxis_tickfont=dict(size=14, family="Times New Roman"),
        #             yaxis_tickfont=dict(size=14, family="Times New Roman"),
        #             zaxis_tickfont=dict(size=14, family="Times New Roman"),
        #             xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
        #             yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
        #             zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
        #         ),
        #         scene_aspectmode='manual',
        #         scene_aspectratio=dict(x=1, y=1, z=.4),
        #         scene_camera=camera,
        #     )
        #     # plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=False)
        #     fig.write_image(filename, width=1980, height=1080)
        pass

    def is_location_legal(self, x, y):
        point = Point(x, y)
        islegal = True
        if self.polygon_obstacle_shapely.contains(point) or not self.polygon_border_shapely.contains(point):
            islegal = False
        return islegal

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
        counter = 0
        xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
        yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
        zrot = -self.gmrf_grid[:, 2]
        self.ind_plot = np.where((zrot < 0) * (zrot >= -5) * (xrot > 70))[0]

        self.yplot = xrot[self.ind_plot]
        self.xplot = yrot[self.ind_plot]
        self.zplot = zrot[self.ind_plot]

        self.dataset_assimilated  = np.empty([0, 3])
        for i in range(0, len(self.lat_auv), AUV_TIMESTEP):
            mu_plot = self.knowledge.mu[self.ind_plot]
            self.var_plot = self.knowledge.SigmaDiag[self.ind_plot]
            self.ep_plot = get_ep(mu_plot.astype(np.float32), self.var_plot.astype(np.float32),
                             np.float32(self.threshold))

            filename = FIGPATH + "mu_cond/jpg/P_{:03d}.jpg".format(counter)
            fig_mu = self.plot_figure(mu_plot, filename, vmin=5, vmax=self.threshold.item(), opacity=.4,
                                      surface_count=10, cmap="BrBG", cbar_title="Salinity")

            filename = FIGPATH + "std_cond/jpg/P_{:03d}.jpg".format(counter)
            fig_std = self.plot_figure(np.sqrt(self.var_plot), filename, vmin=0, vmax=1, opacity=.4,
                                      surface_count=10, cmap="RdBu", cbar_title="STD")

            filename = FIGPATH + "ep_cond/jpg/P_{:03d}.jpg".format(counter)
            fig_ep = self.plot_figure(self.ep_plot, filename, vmin=0, vmax=1, opacity=.4,
                                      surface_count=10, cmap="Brwnyl", cbar_title="EP")

            counter += 1
            print(counter)
            blockPrint()
            if i+AUV_TIMESTEP <= len(self.lat_auv):
                ind_start = i
                ind_end = i + AUV_TIMESTEP
            else:
                ind_start = i
                ind_end = -1

            if i <= IND_END_PRERUN:
                x_trj, y_trj = latlon2xy(self.lat_auv[:ind_end],
                                         self.lon_auv[:ind_end], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                xtemp = (x_trj * np.cos(ROTATED_ANGLE) - y_trj * np.sin(ROTATED_ANGLE))
                ytemp = (x_trj * np.sin(ROTATED_ANGLE) + y_trj * np.cos(ROTATED_ANGLE))
                ztemp = -self.depth_auv[:ind_end]
            else:
                x_trj, y_trj = latlon2xy(self.lat_auv[IND_END_PRERUN:ind_end],
                                         self.lon_auv[IND_END_PRERUN:ind_end], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                xtemp = (x_trj * np.cos(ROTATED_ANGLE) - y_trj * np.sin(ROTATED_ANGLE))
                ytemp = (x_trj * np.sin(ROTATED_ANGLE) + y_trj * np.cos(ROTATED_ANGLE))
                ztemp = -self.depth_auv[IND_END_PRERUN:ind_end]
            self.trajectory_plot = np.vstack((ytemp, xtemp, ztemp)).T

            x, y = latlon2xy(self.lat_auv[ind_start:ind_end],
                             self.lon_auv[ind_start:ind_end], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
            dataset = np.vstack((x, y,
                                 self.depth_auv[ind_start:ind_end],
                                 self.salinity_auv[ind_start:ind_end])).T
            self.ind_measured, self.measurements, self.std_measurements = self.assimilate_data(dataset)

            dataset = np.hstack((vectorise(self.ind_measured), self.measurements, self.std_measurements))
            self.dataset_assimilated = np.append(self.dataset_assimilated, dataset, axis=0)
            # self.sampling_location_plot = []
            # for i in range(len(self.ind_measured)):
            #     xtemp = (self.gmrf_grid[self.ind_measured[i], 0] * np.cos(ROTATED_ANGLE) -
            #              self.gmrf_grid[self.ind_measured[i], 1] * np.sin(ROTATED_ANGLE))
            #     ytemp = (self.gmrf_grid[self.ind_measured[i], 0] * np.sin(ROTATED_ANGLE) +
            #              self.gmrf_grid[self.ind_measured[i], 1] * np.cos(ROTATED_ANGLE))
            #     ztemp = -self.gmrf_grid[self.ind_measured[i], 2]
            #     self.sampling_location_plot.append([ytemp, xtemp, ztemp])

            # self.gmrf_model.update(rel=measurements, ks=ind_measured)
            # self.update_knowledge()

            enablePrint()

            # plotly.offline.plot(fig_mu, filename=FIGPATH + "mu_cond/html/P_{:03d}.html".format(counter), auto_open=False)
            # plotly.offline.plot(fig_std, filename=FIGPATH + "std_cond/html/P_{:03d}.html".format(counter),
            #                     auto_open=False)
            # plotly.offline.plot(fig_ep, filename=FIGPATH + "ep_cond/html/P_{:03d}.html".format(counter),
            #                     auto_open=False)
            print(counter)
            if counter >= 3:
                break

        # df = pd.DataFrame(self.dataset_assimilated , columns=['ind', 'salinity', 'std'])
        # df.to_csv(FILEPATH + "../Experiments/20220511/data4martin.csv", index=False)
        # print("Dataset is saved successfully!")
        os.system("say finished")

    def plot_figure(self, value, filename, vmin=None, vmax=None, opacity=None, surface_count=None, cmap=None,
                    cbar_title=None):
        points_int, values_int = interpolate_3d(self.xplot, self.yplot, self.zplot, value)
        fig = make_subplots(rows=1, cols=1, specs = [[{'type': 'scene'}]])

        # fig.add_trace(go.Volume(
        #     x=points_int[:, 0],
        #     y=points_int[:, 1],
        #     z=points_int[:, 2],
        #     value=values_int.flatten(),
        #     isomin=vmin,
        #     isomax=vmax,
        #     opacity=opacity,
        #     surface_count=surface_count,
        #     coloraxis="coloraxis",
        #     caps=dict(x_show=False, y_show=False, z_show=False),
        # ),
        #     row=1, col=1
        # )

        fig.add_trace(go.Isosurface(
            x=points_int[:, 0],
            y=points_int[:, 1],
            z=points_int[:, 2],
            value=values_int.flatten(),
            isomin=vmin,
            isomax=vmax,
            opacity=opacity,
            surface_count=surface_count,
            coloraxis="coloraxis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=1
        )

        try:
            trjp = np.array(self.sampling_location_plot)
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
            ),
                row=1, col=1
            )
        except:
            pass

        try:
            trjp = np.array(self.trajectory_plot)
            fig.add_trace(go.Scatter3d(
                x=trjp[:, 0],
                y=trjp[:, 1],
                z=trjp[:, 2],
                mode='lines',
                line=dict(
                    width=1,
                    color='black',
                ),
            ),
                row=1, col=1
            )
        except:
            pass

        fig.update_coloraxes(colorscale=cmap, colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                              tickfont=dict(size=18, family="Times New Roman"),
                                                              title=cbar_title,
                                                              titlefont=dict(size=18, family="Times New Roman")),
                             colorbar_x=.95)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=0, z=.5)
        )
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
                xaxis=dict(range=[np.amin(points_int[:, 0]), np.amax(points_int[:, 0])]),
                yaxis=dict(range=[np.amin(points_int[:, 1]), np.amax(points_int[:, 1])]),
                zaxis=dict(nticks=4, range=[-6, .5], ),
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
        std_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
            std_assimilated[i] = np.std(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return ind_assimilated, vectorise(salinity_assimilated), vectorise(std_assimilated)

    def plot_variogram(self):
        self.load_sinmod_data_along_auv_path(data_exists=True)
        fig = go.Figure(data=[go.Scatter3d(
            x=self.lon_auv,
            y=self.lat_auv,
            z=-self.depth_auv,
            mode='markers',
            marker=dict(
                size=12,
                color=values_es,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )
        )])
        pass


if __name__ == "__main__":
    e = EDA()
    # e.load_sinmod_data(data_exists=True)
    # e.plot_scatter_data()
    # e.plot_sinmod()
    # e.plot_prior()
    # e.plot_recap_mission()
    # e.plot_time_series()
    # e.plot_variogram()
    # e.plot_sinmod_layer()
    # e.save_grid_to_gis()


#%%
# plt.scatter(e.lon_sinmod, e.lat_sinmod, c=e.salinity_sinmod[0, 0, :, :], cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
lat = e.lat_sinmod
lon = e.lon_sinmod
sal = e.salinity_sinmod[0, 0, :, :]

# sal[sal<10] = 0
levels = np.arange(15, 33)
plt.contour(e.lon_sinmod, e.lat_sinmod, sal, levels=levels, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
CS = plt.contour(e.lon_sinmod, e.lat_sinmod, sal, levels=[26.8], colors=('red', ), linestyles=('-',), linewidths=(2,))
CS.collections[0].set_label("Threshold=26.8")
plt.contour(e.lon_sinmod, e.lat_sinmod, sal, levels=[0], colors=('w', ), linestyles=('-',), linewidths=(2,))
plt.contourf(e.lon_sinmod, e.lat_sinmod, sal, levels=levels, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
# plt.gca().clabel(CS, inline=True, fontsize=10)
plt.plot(e.polygon_border[:, 1], e.polygon_border[:, 0], 'k-', linewidth=2)
plt.plot(e.polygon_obstacle[:, 1], e.polygon_obstacle[:, 0], 'k-', linewidth=2)
plt.colorbar()
# plt.legend()
plt.ylim([np.amin(e.polygon_border[:, 0]),np.amax(e.polygon_border[:, 0])])
plt.title("T" + datetime.fromtimestamp(e.timestamp[0]).strftime("%Y-%m-%d, %H:%M"))
plt.show()
print("f")
os.system('say finished')

#%%
g = np.array(e.grid_plot)
t = np.array(e.sal_plot)
ind_il = np.where(e.ind_illegal == 1)[0]

t[:, 0, ind_il] = 0

#%%

import plotly.graph_objects as go
import numpy as np

point_es, values_es = interpolate_3d(g[:, 0], g[:, 1], g[:, 2], t[0, 0, :], refill_nan=True)
# point_es = e.points_es
# values_es = e.values_es

vmin = None
vmax = None
opacity = None
surface_count = None
vmin = 10
vmax = 30
opacity = .4
surface_count = 4

fig = go.Figure(data=[go.Volume(
    x=point_es[:, 0],
    y=point_es[:, 1],
    z=point_es[:, 2],
    value=values_es.flatten(),
    isomin=vmin,
    isomax=vmax,
    opacity=opacity,
    surface_count=surface_count,
    coloraxis="coloraxis",
    caps=dict(x_show=False, y_show=False, z_show=False),
)])

# fig = go.Figure(data=[go.Scatter3d(
#     x=point_es[:, 0],
#     y=point_es[:, 1],
#     z=point_es[:, 2],
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=values_es,                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8
#     )
# )])

fig.update_coloraxes(colorscale="Blues", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                      tickfont=dict(size=18, family="Times New Roman"),
                                                      title="EP",
                                                      titlefont=dict(size=18, family="Times New Roman")),
                     colorbar_x=.95)
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.25, y=0, z=.5)
)
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
        zaxis=dict(nticks=4, range=[-6, .5], ),
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

plotly.offline.plot(fig, filename=FIGPATH + "SINMOD.html", auto_open=True)

#%%
grid_data = []
for i in range(e.salinity_sinmod.shape[2]):
    for j in range(e.salinity_sinmod.shape[3]):
        for k in range(e.salinity_sinmod.shape[1]):
            grid_data.append([e.lat_sinmod[i, j], e.lon_sinmod[i, j], e.depth_sinmod[k], e.salinity_sinmod[0, k, i, j]])
gd = np.array(grid_data)

#%%

vmin = None
vmax = None
opacity = None
surface_count = None
vmin = 0
vmax = 30
opacity = .4
surface_count = 4

# fig = go.Figure(data=[go.Volume(
#     x=gd[:, 1],
#     y=gd[:, 0],
#     z=-gd[:, 2],
#     value=gd[:, 3],
#     isomin=vmin,
#     isomax=vmax,
#     opacity=opacity,
#     surface_count=surface_count,
#     coloraxis="coloraxis",
#     caps=dict(x_show=False, y_show=False, z_show=False),
# )])

fig = go.Figure(data=[go.Scatter3d(
    x=point_es[:, 0],
    y=point_es[:, 1],
    z=point_es[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color=values_es,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8,
        cmin=vmin,
        cmax=vmax,
        showscale=True,
    )
)])

fig.update_coloraxes(colorscale="Blues", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                      tickfont=dict(size=18, family="Times New Roman"),
                                                      title="EP",
                                                      titlefont=dict(size=18, family="Times New Roman")),
                     colorbar_x=.95)
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.25, y=0, z=.5)
)
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
        zaxis=dict(nticks=4, range=[-6, .5], ),
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

plotly.offline.plot(fig, filename=FIGPATH + "SINMOD.html", auto_open=True)


#%%
self = e
def is_location_valid(lat, lon):
    isvalid = 1
    point = Point(lat, lon)
    if self.polygon_obstacle_shapely.contains(point) or not self.polygon_border_shapely.contains(point):
        isvalid = 0
    return isvalid

valid_matrix = np.ones_like(lat)
for i in range(valid_matrix.shape[0]):
    for j in range(valid_matrix.shape[1]):
        valid_matrix[i, j] = is_location_valid(lat[i, j], lon[i, j])

#%%
ind_row = np.where(valid_matrix == True)[0]
ind_col = np.where(valid_matrix == True)[1]

#%%
plt.contour(e.lon_sinmod[ind_row, ind_col], e.lat_sinmod[ind_row, ind_col], sal[ind_row, ind_col], levels=levels, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
