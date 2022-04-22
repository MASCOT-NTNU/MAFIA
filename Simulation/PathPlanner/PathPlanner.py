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

        # == set up rotational angle to make the plot easier
        self.knowledge.rotated_angle = ROTATED_ANGLE

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
            break
            # KnowledgePlot(knowledge=self.knowledge, vmin=0, vmax=32,
            #               filename=FILEPATH + "fig/myopic3d/P_{:03d}".format(i), html=False)

            # print("test of plotting")


        # KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename=foldername + "Field_{:03d}".format(i), html=False)


if __name__ == "__main__":
    lat_start = 63.451197
    lon_start = 10.411521
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
        self.prepare_coordinates()
        self.simple_plot()

    def prepare_coordinates(self):
        self.ind_remove_top_layer = np.where(self.coordinates[:, 2]>0)[0]
        xgrid = self.coordinates[self.ind_remove_top_layer, 0]
        ygrid = self.coordinates[self.ind_remove_top_layer, 1]
        xrotated = xgrid * np.cos(self.knowledge.rotated_angle) - ygrid * np.sin(self.knowledge.rotated_angle)
        yrotated = xgrid * np.sin(self.knowledge.rotated_angle) + ygrid * np.cos(self.knowledge.rotated_angle)

        self.xplot = yrotated
        self.yplot = xrotated
        self.zplot = self.coordinates[self.ind_remove_top_layer, 2]

        pass

    def simple_plot(self):


        fig = go.Figure(data=[go.Scatter3d(
            x=self.xplot,
            y=self.yplot,
            z=-self.zplot,
            mode='markers',
            marker=dict(
                size=12,
                color=self.knowledge.mu_cond[self.ind_remove_top_layer],  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        )])

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
                zaxis=dict(nticks=4, range=[-10, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=1),
            scene_camera=camera,
        )

        plotly.offline.plot(fig, filename=self.filename + ".html", auto_open=False)


    def plot(self):
        x = self.coordinates[self.ind_remove_top_layer, 1]
        y = self.coordinates[self.ind_remove_top_layer, 0]
        z = self.coordinates[self.ind_remove_top_layer, 2]
        z_layer = np.unique(z)
        number_of_plots = len(z_layer)

        # print(lat.shape)
        # points_mean, values_mean = interpolate_3d(y, x, z, self.knowledge.mu_cond[self.ind_remove_top_layer])
        # points_std, values_std = interpolate_3d(y, x, z, np.sqrt(self.knowledge.Sigma_cond_diag))
        # points_ep, values_ep = interpolate_3d(y, x, z, self.knowledge.excursion_prob[self.ind_remove_top_layer])

        points_mean = self.coordinates
        values_mean = self.knowledge.mu_cond[self.ind_remove_top_layer]

        points_ep = self.coordinates
        values_ep = self.knowledge.excursion_prob[self.ind_remove_top_layer]

        trajectory = []
        for i in range(len(self.knowledge.trajectory)):
            trajectory.append([self.knowledge.trajectory[i].y,
                               self.knowledge.trajectory[i].x,
                               self.knowledge.trajectory[i].z])

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                            subplot_titles=("Updated field", "Updated excursion probability",))
        # fig = make_subplots(rows = 1, cols = 3, specs = [[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        #                     subplot_titles=("Conditional Mean", "Std", "EP"))
        # fig.add_trace(go.Volume(
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=-z,
            # x=points_mean[:, 1],
            # y=points_mean[:, 0],
            # z=-points_mean[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=self.knowledge.mu_cond[self.ind_remove_top_layer],  # set color to an array/list of desired values
                colorscale='BrBG',  # choose a colorscale
                opacity=0.8
            ),
        ),
            row=1, col=1
        )

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            # x=points_ep[:, 1],
            # y=points_ep[:, 0],
            # z=-points_ep[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=self.knowledge.excursion_prob[self.ind_remove_top_layer],  # set color to an array/list of desired values
                # colorscale='gnbu',  # choose a colorscale
                opacity=0.8
            ),
        ),
            row=1, col=2,
        )

        # if len(self.knowledge.ind_neighbour_filtered_waypoint):
        #     fig.add_trace(go.Scatter3d(
        #         x=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 1],
        #         y=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 0],
        #         z=-self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=15,
        #             color="white",
        #             showscale=False,
        #         ),
        #         showlegend=False,
        #     ),
        #         row='all', col='all'
        #     )

        # if trajectory:
        #     trajectory = np.array(trajectory)
        #     fig.add_trace(go.Scatter3d(
        #         # print(trajectory),
        #         x=trajectory[:, 0],
        #         y=trajectory[:, 1],
        #         z=-trajectory[:, 2],
        #         mode='markers+lines',
        #         marker=dict(
        #             size=5,
        #             color="black",
        #             showscale=False,
        #         ),
        #         line=dict(
        #             color="yellow",
        #             width=3,
        #             showscale=False,
        #         ),
        #         showlegend=False,
        #     ),
        #         row='all', col='all'
        #     )
        #
        # fig.add_trace(go.Scatter3d(
        #     x=[self.knowledge.current_location.y],
        #     y=[self.knowledge.current_location.x],
        #     z=[-self.knowledge.current_location.z],
        #     mode='markers',
        #     marker=dict(
        #         size=20,
        #         color="red",
        #         showscale=False,
        #     ),
        #     showlegend=False,  # remove all unnecessary trace names
        # ),
        #     row='all', col='all'
        # )

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
                zaxis=dict(nticks=4, range=[-10, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene2=dict(
                zaxis=dict(nticks=4, range=[-10, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene2_aspectmode='manual',
            scene2_aspectratio=dict(x=1, y=1, z=.5),
            scene3=dict(
                zaxis=dict(nticks=4, range=[-10, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
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



#%%
grid = np.load(FILEPATH+"models/grid.npy")
lats = np.load(FILEPATH+"models/lats.npy")
lons = np.load(FILEPATH+"models/lons.npy")
depth = np.load(FILEPATH+"models/depth.npy")

lat_box = grid[:, 2]
lon_box = grid[:, 3]

lat_origin = lat_box[0]
lon_origin = lon_box[0]

x, y = latlon2xy(lat_box, lon_box, lat_origin, lon_origin)
angle = np.math.atan2(x[1] - x[0], y[1] - y[0])
import math

plt.plot(y, x)
plt.plot(y[0], x[0], 'o')
plt.plot(y[1], x[1], 'x')
plt.show()
# angle = np.math.atan2(lat_box[1] - lat_box[0], lon_box[1] - lon_box[0])
angle = np.math.atan2(x[1] - x[0], y[1] - y[0])
print("Angle is: ", math.degrees(angle))

plt.figure(figsize=(10, 10))
# plt.plot(lons, lats, 'k.')
# plt.show()


xgrid, ygrid = latlon2xy(lats, lons, lat_origin, lon_origin)
plt.plot(ygrid, xgrid, 'g.')
plt.show()

#%%
# angle = math.radians(30)
xp = xgrid * np.cos(angle) - ygrid * np.sin(angle)
yp = xgrid * np.sin(angle) + ygrid * np.cos(angle)

latsp, lonsp = xy2latlon(xp, yp, lat_origin, lon_origin)

plt.plot(lonsp, latsp, 'r.')
plt.show()

