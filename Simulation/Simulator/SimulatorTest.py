"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""
import time

import numpy as np

from usr_func import *
from MAFIA.Simulation.Config.Config import *
from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Knowledge.Knowledge import Knowledge
from MAFIA.spde import spde
import pickle

# == Set up
LAT_START = 63.447231
LON_START = 10.412948
DEPTH_START = .5
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
Z_START = DEPTH_START
VERTICES_TRANSECT = np.array([[63.450421, 10.395289],
                              [63.453768, 10.420457],
                              [63.446442, 10.412006]])
DEPTH_TOP = .5
DEPTH_BOTTOM = 5.5
YOYO_LATERAL_DISTANCE = 60
# ==


class Simulator:

    def __init__(self):
        self.load_waypoint()
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.load_prior()
        self.load_simulated_truth()
        self.update_knowledge()
        self.load_hash_neighbours()
        self.load_hash_waypoint2gmrf()
        self.initialise_function_calls()
        print("S1-S9 complete!")

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Simulation/Config/WaypointGraph.csv").to_numpy()
        print("S1: Waypoint is loaded successfully!")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Simulation/Config/GMRFGrid.csv").to_numpy()
        self.N_gmrf_grid = len(self.gmrf_grid)
        print("S2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2, reduce=True, method=2)
        print("S3: GMRF model is loaded successfully!")

    def load_prior(self):
        print("S4: Prior is loaded successfully!")
        pass

    def load_simulated_truth(self):
        path_mu_truth = FILEPATH + "Simulation/Config/Data/data_mu_truth.csv"
        self.simulated_truth = pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1)
        print("S5: Simulated truth is loaded successfully!")

    def update_knowledge(self):
        self.knowledge = Knowledge(gmrf_grid=self.gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
        print("S6: Knowledge of the field is set up successfully!")

    def load_hash_neighbours(self):
        neighbour_file = open(FILEPATH + "Simulation/Config/HashNeighbours.p", 'rb')
        self.hash_neighbours = pickle.load(neighbour_file)
        neighbour_file.close()
        print("S7: Neighbour hash table is loaded successfully!")

    def load_hash_waypoint2gmrf(self):
        waypoint2gmrf_file = open(FILEPATH + "Simulation/Config/HashWaypoint2GMRF.p", 'rb')
        self.hash_waypoint2gmrf = pickle.load(waypoint2gmrf_file)
        waypoint2gmrf_file.close()
        print("S8: Waypoint2GMRF hash table is loaded successfully!")

    def initialise_function_calls(self):
        get_ind_at_location3d_xyz(self.waypoints, 1, 2, 3)  # used to initialise the function call
        print("S9: Function calls are initialised successfully!")

    def get_transect_trajectory(self):
        self.trajectory_transect = []
        for i in range(len(VERTICES_TRANSECT)-1):
            lat_start, lon_start = VERTICES_TRANSECT[i, :]
            lat_end, lon_end = VERTICES_TRANSECT[i+1, :]
            x_range, y_range = latlon2xy(lat_end, lon_end, lat_start, lon_start)
            distance = np.sqrt(x_range**2 + y_range**2)
            gap = np.arange(0, distance, YOYO_LATERAL_DISTANCE)
            angle = np.math.atan2(x_range, y_range)
            x_loc = gap * np.sin(angle)
            y_loc = gap * np.cos(angle)
            for j in range(len(x_loc)):
                if isEven(j):
                    lat_up, lon_up = xy2latlon(x_loc[j], y_loc[j], lat_start, lon_start)
                    self.trajectory_transect.append([lat_up, lon_up, DEPTH_TOP])
                else:
                    lat_down, lon_down = xy2latlon(x_loc[j], y_loc[j], lat_start, lon_start)
                    self.trajectory_transect.append([lat_down, lon_down, DEPTH_BOTTOM])
        t = np.array(self.trajectory_transect)
        fig = go.Figure(data=go.Scatter3d(
            x = t[:, 1],
            y = t[:, 0],
            z = -t[:, 2],
        ))
        plotly.offline.plot(fig, filename=FIGPATH+"transect_line.html", auto_open=True)

    def run(self):
        ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, X_START, Y_START, Z_START)
        ind_previous_waypoint = ind_current_waypoint
        ind_pioneer_waypoint = ind_current_waypoint
        ind_next_waypoint = ind_current_waypoint
        ind_visited_waypoint = []
        ind_visited_waypoint.append(ind_current_waypoint)
        for i in range(NUM_STEPS):
            print("Step: ", i)
            ind_sample_gmrf = self.hash_waypoint2gmrf[ind_current_waypoint]
            self.salinity_measured = self.simulated_truth[ind_sample_gmrf][0]

            t1 = time.time()
            self.gmrf_model.update(rel=self.salinity_measured, ks=ind_sample_gmrf)
            t2 = time.time()
            print("Update consumed: ", t2 - t1)

            self.knowledge.mu = self.gmrf_model.mu
            self.knowledge.SigmaDiag = self.gmrf_model.mvar()

            if i == 0:
                self.pathplanner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints,
                                                    gmrf_model=self.gmrf_model,
                                                    ind_current=ind_current_waypoint,
                                                    ind_previous=ind_previous_waypoint,
                                                    hash_neighbours=self.hash_neighbours,
                                                    hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                                    ind_visited=ind_visited_waypoint)
                ind_next_waypoint = self.pathplanner.ind_next
                self.pathplanner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints,
                                                    gmrf_model=self.gmrf_model,
                                                    ind_current=ind_next_waypoint,
                                                    ind_previous=ind_current_waypoint,
                                                    hash_neighbours=self.hash_neighbours,
                                                    hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                                    ind_visited=ind_visited_waypoint)
                ind_pioneer_waypoint = self.pathplanner.ind_next
            else:
                self.pathplanner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints,
                                                    gmrf_model=self.gmrf_model,
                                                    ind_current=ind_next_waypoint,
                                                    ind_previous=ind_current_waypoint,
                                                    hash_neighbours=self.hash_neighbours,
                                                    hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                                    ind_visited=ind_visited_waypoint)
                ind_pioneer_waypoint = self.pathplanner.ind_next

            # == plot gmrf section
            xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
            yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
            zrot = -self.gmrf_grid[:, 2]

            ind_plot = np.where((zrot<0) * (zrot>=-5) * (xrot>70))[0]
            mu_plot = self.knowledge.mu[ind_plot]
            self.yplot = xrot[ind_plot]
            self.xplot = yrot[ind_plot]
            self.zplot = zrot[ind_plot]

            # fig = go.Figure(data=go.Scatter3d(
            #     x=self.xplot,
            #     y=self.yplot,
            #     z=self.zplot,
            #     mode='markers',
            #     marker=dict(color=mu_plot, size=2, opacity=.0)
            # ))
            # fig.add_trace(go.Scatter3d(
            #     x=self.xplot,
            #     y=self.yplot,
            #     z=self.zplot,
            #     mode='markers',
            #     marker=dict(color=mu_plot)
            # ))
            # fig.add_trace(go.Scatter3d(
            #     x=self.waypoints[:, 1],
            #     y=self.waypoints[:, 0],
            #     z=-self.waypoints[:, 2],
            #     mode='markers',
            #     marker=dict(color='black', size=1, opacity=.1)
            # ))

            points_int, values_int = interpolate_3d(self.xplot, self.yplot, self.zplot, mu_plot)
            fig = go.Figure(data=go.Volume(
                x=points_int[:, 0],
                y=points_int[:, 1],
                z=points_int[:, 2],
                value=values_int.flatten(),
                # isomin=self.vmin,
                # isomax=self.vmax,
                opacity=.4,
                surface_count=10,
                coloraxis="coloraxis",
                caps=dict(x_show=False, y_show=False, z_show=False),
            ),
            )

            # == plot waypoint section
            xrot = self.waypoints[:, 0] * np.cos(ROTATED_ANGLE) - self.waypoints[:, 1] * np.sin(ROTATED_ANGLE)
            yrot = self.waypoints[:, 0] * np.sin(ROTATED_ANGLE) + self.waypoints[:, 1] * np.cos(ROTATED_ANGLE)
            zrot = -self.waypoints[:, 2]
            fig.add_trace(go.Scatter3d(
                x=yrot,
                y=xrot,
                z=zrot,
                mode='markers',
                marker=dict(color='black', size=1, opacity=.1)
            ))
            fig.add_trace(go.Scatter3d(
                x=[yrot[ind_previous_waypoint]],
                y=[xrot[ind_previous_waypoint]],
                z=[zrot[ind_previous_waypoint]],
                mode='markers',
                marker=dict(color='yellow', size=10)
            ))
            fig.add_trace(go.Scatter3d(
                x=[yrot[ind_current_waypoint]],
                y=[xrot[ind_current_waypoint]],
                z=[zrot[ind_current_waypoint]],
                mode='markers',
                marker=dict(color='red', size=10)
            ))
            fig.add_trace(go.Scatter3d(
                x=[yrot[ind_next_waypoint]],
                y=[xrot[ind_next_waypoint]],
                z=[zrot[ind_next_waypoint]],
                mode='markers',
                marker=dict(color='blue', size=10)
            ))
            fig.add_trace(go.Scatter3d(
                x=[yrot[ind_pioneer_waypoint]],
                y=[xrot[ind_pioneer_waypoint]],
                z=[zrot[ind_pioneer_waypoint]],
                mode='markers',
                marker=dict(color='green', size=10)
            ))
            fig.add_trace(go.Scatter3d(
                x=yrot[ind_visited_waypoint],
                y=xrot[ind_visited_waypoint],
                z=zrot[ind_visited_waypoint],
                mode='markers+lines',
                marker=dict(color='black', size=4),
                line=dict(color='black', width=3)
            ))
            fig.add_trace(go.Scatter3d(
                x=yrot[self.pathplanner.ind_candidates],
                y=xrot[self.pathplanner.ind_candidates],
                z=zrot[self.pathplanner.ind_candidates],
                mode='markers',
                marker=dict(color='orange', size=5, opacity=.3)
            ))
            fig.update_coloraxes(colorscale="BrBG", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                                  tickfont=dict(size=18, family="Times New Roman"),
                                                                  title="Salinity",
                                                                  titlefont=dict(size=18, family="Times New Roman")))
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.25, y=-1.25, z=.5)
            )
            fig.update_layout(coloraxis_colorbar_x=0.8)
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
                    zaxis=dict(nticks=4, range=[-5, -0.5], ),
                    xaxis_tickfont=dict(size=14, family="Times New Roman"),
                    yaxis_tickfont=dict(size=14, family="Times New Roman"),
                    zaxis_tickfont=dict(size=14, family="Times New Roman"),
                    xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                    yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                    zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
                ),
                scene_aspectmode='manual',
                scene_aspectratio=dict(x=1, y=1, z=.25),
                scene_camera=camera,
            )

            # plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=False)
            fig.write_image(FIGPATH+"myopic3d/P_{:03d}.jpg".format(i), width=1980, height=1080)

            ind_previous_waypoint = ind_current_waypoint
            ind_current_waypoint = ind_next_waypoint
            ind_next_waypoint = ind_pioneer_waypoint
            ind_visited_waypoint.append(ind_current_waypoint)
            print("previous ind: ", ind_previous_waypoint)
            print("current ind: ", ind_current_waypoint)
            print("next ind: ", ind_next_waypoint)
            print("pioneer ind: ", ind_pioneer_waypoint)

            if i == NUM_STEPS-1:
                plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=True)
            # plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=True)
            break

    def check_assimilation(self):
        print("hello world")
        x_start = 1000
        y_start = -500
        z_start = .5
        x_end = 500
        y_end = 0
        z_end = 5.5
        N = 20
        x = np.linspace(x_start, x_end, N)
        y = np.linspace(y_start, y_end, N)
        z = np.linspace(z_start, z_end, N)
        dataset = np.vstack((x, y, z, np.zeros_like(z))).T
        ind = self.assimilate_data(dataset)

        fig = go.Figure(data=go.Scatter3d(
            x=self.gmrf_grid[:, 1],
            y=self.gmrf_grid[:, 0],
            z=-self.gmrf_grid[:, 2],
            mode='markers',
            marker=dict(color='black', size=2, opacity=.5)
        ))
        fig.add_trace(go.Scatter3d(
            x=y,
            y=x,
            z=-z,
            mode='lines+markers',
            marker=dict(color='red', size=10, opacity=.5),
            line=dict(color='red', width=4)
        ))
        fig.add_trace(go.Scatter3d(
            x=self.gmrf_grid[ind, 1],
            y=self.gmrf_grid[ind, 0],
            z=-self.gmrf_grid[ind, 2],
            mode='markers',
            marker=dict(color='blue', size=20, opacity=.5)
        ))
        plotly.offline.plot(fig, filename=FIGPATH + "check_assimilation.html", auto_open=True)
        pass


    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[:10, :])
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) <= .25)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        print("dataset after filtering: ", dataset[:10, :])
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
    s = Simulator()
    # s.get_transect_trajectory()
    # s.run()
    s.check_assimilation()

#%%
plt.plot(s.gmrf_grid[:, 1], s.gmrf_grid[:, 0], 'k.')
plt.show()




