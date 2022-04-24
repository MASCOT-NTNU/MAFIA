"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""

from usr_func import *
from MAFIA.Simulation.Config.Config import *
from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Knowledge.Knowledge import Knowledge
from MAFIA.spde import spde
import pickle

# == Set up
LAT_START = 63.448747,
LON_START = 10.416038
DEPTH_START = .5
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
Z_START = DEPTH_START
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
        print("S2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2)
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

    def run(self):
        ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, X_START, Y_START, Z_START)
        ind_previous_waypoint = ind_current_waypoint
        ind_visited = []
        ind_visited.append(ind_current_waypoint)
        for i in range(NUM_STEPS):
            print("Step: ", i)
            ind_sample = self.hash_waypoint2gmrf[ind_current_waypoint]
            self.salinity_measured = self.simulated_truth[ind_sample][0]

            t1 = time.time()
            self.gmrf_model.update(rel=self.salinity_measured, ks=ind_sample)
            t2 = time.time()
            print("Update consumed: ", t2 - t1)

            self.knowledge.mu = self.gmrf_model.mu
            self.knowledge.SigmaDiag = self.gmrf_model.mvar()

            planner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints, gmrf_model=self.gmrf_model,
                                       ind_current=ind_current_waypoint, ind_previous=ind_previous_waypoint,
                                       hash_neighbours=self.hash_neighbours, hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                       ind_visited=ind_visited)

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
            #     marker=dict(color='black', size=2, opacity=.0)
            # ))
            # fig.add_trace(go.Scatter3d(
            #     x=self.xplot,
            #     y=self.yplot,
            #     z=self.zplot,
            #     mode='markers',
            #     marker=dict(color=mu_plot)
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
            # fig.add_trace(go.Scatter3d(
            #     x=self.waypoints[:, 1],
            #     y=self.waypoints[:, 0],
            #     z=-self.waypoints[:, 2],
            #     mode='markers',
            #     marker=dict(color='black', size=1, opacity=.1)
            # ))

            # == plot waypoint section
            xrot = self.waypoints[:, 0] * np.cos(ROTATED_ANGLE) - self.waypoints[:, 1] * np.sin(ROTATED_ANGLE)
            yrot = self.waypoints[:, 0] * np.sin(ROTATED_ANGLE) + self.waypoints[:, 1] * np.cos(ROTATED_ANGLE)
            zrot = -self.waypoints[:, 2]
            # fig.add_trace(go.Scatter3d(
            #     x=yrot,
            #     y=xrot,
            #     z=zrot,
            #     mode='markers',
            #     marker=dict(color='black', size=2, opacity=.1)
            # ))
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
                x=[yrot[ind_visited]],
                y=[xrot[ind_visited]],
                z=[zrot[ind_visited]],
                mode='markers+lines',
                marker=dict(color='black', size=4),
                line=dict(color='black')
            ))
            fig.add_trace(go.Scatter3d(
                x=[yrot[planner.ind_next]],
                y=[xrot[planner.ind_next]],
                z=[zrot[planner.ind_next]],
                mode='markers',
                marker=dict(color='blue', size=10)
            ))
            fig.add_trace(go.Scatter3d(
                x=yrot[planner.ind_candidates],
                y=xrot[planner.ind_candidates],
                z=zrot[planner.ind_candidates],
                mode='markers',
                marker=dict(color='green', size=5, opacity=.7)
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

            plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=False)

            ind_previous_waypoint = ind_current_waypoint
            ind_current_waypoint = planner.ind_next
            ind_visited.append(ind_current_waypoint)
            print("previous ind: ", ind_previous_waypoint)
            print("current ind: ", ind_current_waypoint)
            os.system('say finished')
            if i == 20:
                break
        pass

if __name__ == "__main__":
    s = Simulator()
    s.run()





