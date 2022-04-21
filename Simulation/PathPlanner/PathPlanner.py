"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from usr_func import *
from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Simulator.Sampler import Sampler
from MAFIA.Simulation.Field.Grid.Location import Location
from MAFIA.Simulation.Field.Knowledge.Knowledge import Knowledge
from MAFIA.Simulation.Plotting.KnowledgePlot import KnowledgePlot
from MAFIA.Simulation.Config.Config import *
from MAFIA.spde import spde
import pickle


class PathPlanner:

    def __init__(self, starting_location=None):
        self.starting_location = starting_location
        self.setup_pathplanner()

    def setup_pathplanner(self):
        self.spde_model = spde(model=2)
        self.load_spde_grid()
        self.load_waypoint()
        self.initialise_function_calls()
        self.load_ground_truth()
        self.setup_knowledge()
        self.num_steps = NUM_STEPS

    def load_spde_grid(self):
        # sal = np.load(FILEPATH + "models/prior.npy")
        lats = np.load(FILEPATH + "models/lats.npy")
        lons = np.load(FILEPATH + "models/lons.npy")
        depth = np.load(FILEPATH + "models/depth.npy")
        x, y = latlon2xy(lats, lons, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        z = depth
        self.coordinates_spde_xyz = np.vstack((x, y, z)).T
        print("SPDE grid is loaded successfully!")

    def load_waypoint(self):
        self.coordinates_waypoint = pd.read_csv(FILEPATH + "Simulation/Field/Grid/Grid.csv").to_numpy()
        self.waypoint_xyz = self.coordinates_waypoint[:, -3:]
        neighbour_hash_table_filehandler = open(FILEPATH + "Simulation/Field/Grid/Neighbours.p", 'rb')
        self.neighbour_hash_table_waypoint = pickle.load(neighbour_hash_table_filehandler)
        neighbour_hash_table_filehandler.close()
        print("Waypoint is loaded successfully!")

    def initialise_function_calls(self):
        get_ind_at_location3d_xyz(self.waypoint_xyz, 1, 2, 3)  # used to initialise the function call
        print("Function calls are initialised successfully! Enjoy fast and furious!")

    def load_ground_truth(self):
        path_mu_truth = FILEPATH + "Simulation/Field/Data/data_mu_truth.csv"
        self.ground_truth = pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1)

    def setup_knowledge(self):
        self.knowledge = Knowledge(coordinates_grid=self.coordinates_spde_xyz, coordinates_waypoint=self.waypoint_xyz,
                                   neighbour_hash_table_waypoint=self.neighbour_hash_table_waypoint,
                                   threshold=THRESHOLD, spde_model=self.spde_model,
                                   previous_location=self.starting_location, current_location=self.starting_location)
        self.ind_sample_grid = get_ind_at_location3d_xyz(self.coordinates_spde_xyz,
                                                         self.knowledge.current_location.x,
                                                         self.knowledge.current_location.y,
                                                         self.knowledge.current_location.z)
        print("ind_sample_grid: ", self.ind_sample_grid)
        self.knowledge.ind_current_location_waypoint = get_ind_at_location3d_xyz(self.waypoint_xyz,
                                                                                 self.knowledge.current_location.x,
                                                                                 self.knowledge.current_location.y,
                                                                                 self.knowledge.current_location.z)
        self.knowledge.ind_previous_location_waypoint = self.knowledge.ind_current_location_waypoint
        self.knowledge.next_location = self.knowledge.current_location

    def run(self):
        for i in range(self.num_steps):
            print("Step No. ", i)
            Sampler(self.knowledge, self.ground_truth, self.ind_sample_grid)
            myopic3d = MyopicPlanning3D(knowledge=self.knowledge)
            self.next_location = self.knowledge.next_location

            print("next location: ", self.next_location.x, self.next_location.y, self.next_location.z)
            self.ind_sample_grid = get_ind_at_location3d_xyz(self.coordinates_spde_xyz, self.next_location.x,
                                                             self.next_location.y, self.next_location.z)
            print("ind_sample_grid: ", self.ind_sample_grid)

            self.knowledge.step_no = i
            KnowledgePlot(knowledge=self.knowledge, vmin=0, vmax=32,
                          filename=FILEPATH + "fig/myopic3d/P_{:03d}".format(i), html=False)

            print("test of plotting")

            break
        # KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename=foldername + "Field_{:03d}".format(i), html=False)


if __name__ == "__main__":
    lat_start = 63.45121
    lon_start = 10.40673
    depth_start = .5
    x, y = latlon2xy(lat_start, lon_start, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
    z = depth_start
    starting_location = Location(x, y, z)

    p = PathPlanner(starting_location=starting_location)
    p.run()

#%%

class KnowledgePlot:

    def __init__(self, knowledge=None, vmin=28, vmax=30, filename="mean", html=False):
        if knowledge is None:
            raise ValueError("")
        self.knowledge = knowledge
        self.coordinates = self.knowledge.coordinates_grid
        self.vmin = vmin
        self.vmax = vmax
        self.filename = filename
        self.html = html
        # self.plot()
        self.simple_plot()

    def simple_plot(self):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        z = self.coordinates[:, 2]

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=-z,
            mode='markers',
            marker=dict(
                size=12,
                color=self.knowledge.mu_cond,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        )])

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        plotly.offline.plot(fig, filename=FIGPATH+"test_plot.html", auto_open=True)


    def plot(self):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        z = self.coordinates[:, 2]
        z_layer = np.unique(z)
        number_of_plots = len(z_layer)

        # print(lat.shape)
        points_mean, values_mean = interpolate_3d(y, x, z, self.knowledge.mu_cond)
        # points_std, values_std = interpolate_3d(y, x, z, np.sqrt(self.knowledge.Sigma_cond_diag))
        points_ep, values_ep = interpolate_3d(y, x, z, self.knowledge.excursion_prob)

        trajectory = []
        for i in range(len(self.knowledge.trajectory)):
            trajectory.append([self.knowledge.trajectory[i].x,
                               self.knowledge.trajectory[i].y,
                               self.knowledge.trajectory[i].z])

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                            subplot_titles=("Updated field", "Updated excursion probability",))
        # fig = make_subplots(rows = 1, cols = 3, specs = [[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        #                     subplot_titles=("Conditional Mean", "Std", "EP"))
        fig.add_trace(go.Volume(
            x=points_mean[:, 0],
            y=points_mean[:, 1],
            z=-points_mean[:, 2],
            value=values_mean.flatten(),
            isomin=self.vmin,
            isomax=self.vmax,
            opacity=.1,
            surface_count=30,
            colorscale="BrBG",
            # coloraxis="coloraxis1",
            colorbar=dict(x=0.5, y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=1
        )
        # print(values_std)
        # if len(values_std):
        #     fig.add_trace(go.Volume(
        #         x=points_std[:, 0],
        #         y=points_std[:, 1],
        #         z=-points_std[:, 2],
        #         value=values_std.flatten(),
        #         isomin=0,
        #         isomax=1,
        #         opacity=.1,
        #         surface_count=30,
        #         colorscale = "rdbu",
        #         colorbar=dict(x=0.65, y=0.5, len=.5),
        #         reversescale=True,
        #         caps=dict(x_show=False, y_show=False, z_show=False),
        #     ),
        #         row=1, col=2
        #     )

        fig.add_trace(go.Volume(
            x=points_ep[:, 0],
            y=points_ep[:, 1],
            z=-points_ep[:, 2],
            value=values_ep.flatten(),
            isomin=0,
            isomax=1,
            opacity=.1,
            surface_count=30,
            colorscale="gnbu",
            colorbar=dict(x=1, y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=2,
            # row = 1, col = 3,
        )

        if len(self.knowledge.ind_neighbour_filtered_waypoint):
            fig.add_trace(go.Scatter3d(
                x=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 0],
                y=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 1],
                z=-self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    color="white",
                    showscale=False,
                ),
                showlegend=False,
            ),
                row='all', col='all'
            )

        # if len(self.knowledge.ind_cand_filtered):
        #     fig.add_trace(go.Scatter3d(
        #         x=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 1],
        #         y=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 0],
        #         z=-self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=10,
        #             color="blue",
        #             showscale=False,
        #         ),
        #         showlegend=False, # remove all unnecessary trace names
        #     ),
        #         row='all', col='all'
        #     )

        if trajectory:
            trajectory = np.array(trajectory)
            fig.add_trace(go.Scatter3d(
                # print(trajectory),
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=-trajectory[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color="black",
                    showscale=False,
                ),
                line=dict(
                    color="yellow",
                    width=3,
                    showscale=False,
                ),
                showlegend=False,
            ),
                row='all', col='all'
            )

        fig.add_trace(go.Scatter3d(
            x=[self.knowledge.current_location.x],
            y=[self.knowledge.current_location.y],
            z=[-self.knowledge.current_location.z],
            mode='markers',
            marker=dict(
                size=20,
                color="red",
                showscale=False,
            ),
            showlegend=False,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.25, y=2.25, z=2.25)
        )

        fig.update_layout(
            title={
                'text': "Simulation",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-3, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene2=dict(
                zaxis=dict(nticks=4, range=[-3, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene2_aspectmode='manual',
            scene2_aspectratio=dict(x=1, y=1, z=.5),
            scene3=dict(
                zaxis=dict(nticks=4, range=[-3, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene3_aspectmode='manual',
            scene3_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
            scene2_camera=camera,
            scene3_camera=camera,
        )

        # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        # if self.html:
        # print("Save html")
        plotly.offline.plot(fig, filename=self.filename + ".html", auto_open=False)
        # os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html")
        # fig.write_image(self.filename+".png", width=1980, height=1080, engine = "orca")


self = p
kp = KnowledgePlot(knowledge=self.knowledge, vmin=0, vmax=32,
              filename=FILEPATH + "fig/myopic3d/P_{:03d}".format(0), html=False)







#%%

# import plotly.graph_objects as go
# import numpy as np
# import plotly
# # Helix equation
#
# fig = go.Figure(data=[go.Scatter3d(
#     x=lons,
#     y=lats,
#     z=depth,
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=sal,                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8,
#         cmin=0,
#         cmax=30,
#     )
#
# )])
#
# # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# plotly.offline.plot(fig, filename=FIGPATH+"prior.html", auto_open=True)




