"""
This script plots the knowledge
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-22
"""


from usr_func import *
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import os


class KnowledgePlot:

    def __init__(self, knowledge=None, vmin=None, vmax=None, filename="mean", html=False):
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
        self.plot()

    def prepare_coordinates(self):
        self.ind_remove_top_layer = np.where(self.coordinates[:, 2]>0)[0]
        xgrid = self.coordinates[self.ind_remove_top_layer, 0]
        ygrid = self.coordinates[self.ind_remove_top_layer, 1]
        xrotated = xgrid * np.cos(self.knowledge.rotated_angle) - ygrid * np.sin(self.knowledge.rotated_angle)
        yrotated = xgrid * np.sin(self.knowledge.rotated_angle) + ygrid * np.cos(self.knowledge.rotated_angle)

        self.xplot = yrotated
        self.yplot = xrotated

        pass

    def plot(self):
        self.knowledge.mu_cond[self.knowledge.mu_cond<0] = 0
        self.ind_selected_to_plot = np.where(self.knowledge.mu_cond[self.ind_remove_top_layer]>=0)[0]
        self.xplot = self.xplot[self.ind_selected_to_plot]
        self.yplot = self.yplot[self.ind_selected_to_plot]
        self.zplot = -self.coordinates[self.ind_remove_top_layer, 2][self.ind_selected_to_plot]

        points_mean, values_mean = interpolate_3d(self.xplot, self.yplot, self.zplot,
                                                  self.knowledge.mu_cond[self.ind_remove_top_layer][self.ind_selected_to_plot])

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        fig.add_trace(go.Volume(
            x=points_mean[:, 0],
            y=points_mean[:, 1],
            z=points_mean[:, 2],
            value=values_mean,
            isomin=self.vmin,
            isomax=self.vmax,
            opacity=.3,
            surface_count=10,
            colorscale="BrBG",
            # coloraxis="coloraxis1",
            # colorbar=dict(x=0.5, y=0.5, len=.5),
            # reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        # fig.add_trace(go.Scatter3d(
        #     x=self.xplot,
        #     y=self.yplot,
        #     z=self.zplot,
        #     mode='markers',
        #     marker=dict(
        #         size=12,
        #         color=self.knowledge.mu_cond[self.ind_remove_top_layer][self.ind_selected_to_plot],  # set color to an array/list of desired values
        #         colorscale='BrBG',  # choose a colorscale
        #         showscale=True,
        #         opacity=0.1
        #     )
        # ))

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

        fig.add_trace(go.Scatter3d(
            x=[self.knowledge.current_location.Y_START],
            y=[self.knowledge.current_location.X_START],
            z=[-self.knowledge.current_location.Z_START],
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

        fig.add_trace(go.Scatter3d(
            x=[self.knowledge.next_location.Y_START],
            y=[self.knowledge.next_location.X_START],
            z=[-self.knowledge.next_location.Z_START],
            mode='markers',
            marker=dict(
                size=20,
                color="blue",
                showscale=False,
            ),
            showlegend=False,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        trajectory = []
        for i in range(len(self.knowledge.sampling_location_plot)):
            trajectory.append([self.knowledge.sampling_location_plot[i].Y_START,
                               self.knowledge.sampling_location_plot[i].X_START,
                               self.knowledge.sampling_location_plot[i].Z_START])

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


        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
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
            scene_camera=camera,
        )

        plotly.offline.plot(fig, filename=self.filename + ".html", auto_open=False)



# class KnowledgePlot:
#
#     def __init__(self, knowledge=None, vmin=28, vmax=30, filename="mean", html=False):
#         if knowledge is None:
#             raise ValueError("")
#         self.knowledge = knowledge
#         # self.coordinates = self.knowledge.coordinates_waypoint
#         self.coordinates = self.knowledge.coordinates_grid
#         self.vmin = vmin
#         self.vmax = vmax
#         self.filename = filename
#         self.html = html
#         self.plot()
#
#     def plot(self):
#         x = self.coordinates[:, 0]
#         y = self.coordinates[:, 1]
#         z = self.coordinates[:, 2]
#         z_layer = np.unique(z)
#         number_of_plots = len(z_layer)
#
#         # print(lat.shape)
#         points_mean, values_mean = interpolate_3d(y, x, z, self.knowledge.mu_cond)
#         # points_std, values_std = interpolate_3d(y, x, z, np.sqrt(self.knowledge.Sigma_cond_diag))
#         points_ep, values_ep = interpolate_3d(y, x, z, self.knowledge.excursion_prob)
#
#         trajectory = []
#         for i in range(len(self.knowledge.trajectory)):
#             trajectory.append([self.knowledge.trajectory[i].x,
#                                self.knowledge.trajectory[i].y,
#                                self.knowledge.trajectory[i].z])
#
#         fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
#                             subplot_titles=("Updated field", "Updated excursion probability",))
#         # fig = make_subplots(rows = 1, cols = 3, specs = [[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
#         #                     subplot_titles=("Conditional Mean", "Std", "EP"))
#         fig.add_trace(go.Volume(
#             x=points_mean[:, 0],
#             y=points_mean[:, 1],
#             z=-points_mean[:, 2],
#             value=values_mean.flatten(),
#             isomin=self.vmin,
#             isomax=self.vmax,
#             opacity=.1,
#             surface_count=30,
#             colorscale="BrBG",
#             # coloraxis="coloraxis1",
#             colorbar=dict(x=0.5, y=0.5, len=.5),
#             reversescale=True,
#             caps=dict(x_show=False, y_show=False, z_show=False),
#         ),
#             row=1, col=1
#         )
#         # print(values_std)
#         # if len(values_std):
#         #     fig.add_trace(go.Volume(
#         #         x=points_std[:, 0],
#         #         y=points_std[:, 1],
#         #         z=-points_std[:, 2],
#         #         value=values_std.flatten(),
#         #         isomin=0,
#         #         isomax=1,
#         #         opacity=.1,
#         #         surface_count=30,
#         #         colorscale = "rdbu",
#         #         colorbar=dict(x=0.65, y=0.5, len=.5),
#         #         reversescale=True,
#         #         caps=dict(x_show=False, y_show=False, z_show=False),
#         #     ),
#         #         row=1, col=2
#         #     )
#
#         fig.add_trace(go.Volume(
#             x=points_ep[:, 0],
#             y=points_ep[:, 1],
#             z=-points_ep[:, 2],
#             value=values_ep.flatten(),
#             isomin=0,
#             isomax=1,
#             opacity=.1,
#             surface_count=30,
#             colorscale="gnbu",
#             colorbar=dict(x=1, y=0.5, len=.5),
#             reversescale=True,
#             caps=dict(x_show=False, y_show=False, z_show=False),
#         ),
#             row=1, col=2,
#             # row = 1, col = 3,
#         )
#
#         if len(self.knowledge.ind_neighbour_filtered_waypoint):
#             fig.add_trace(go.Scatter3d(
#                 x=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 0],
#                 y=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 1],
#                 z=-self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 2],
#                 mode='markers',
#                 marker=dict(
#                     size=15,
#                     color="white",
#                     showscale=False,
#                 ),
#                 showlegend=False,
#             ),
#                 row='all', col='all'
#             )
#
#         # if len(self.knowledge.ind_cand_filtered):
#         #     fig.add_trace(go.Scatter3d(
#         #         x=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 1],
#         #         y=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 0],
#         #         z=-self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 2],
#         #         mode='markers',
#         #         marker=dict(
#         #             size=10,
#         #             color="blue",
#         #             showscale=False,
#         #         ),
#         #         showlegend=False, # remove all unnecessary trace names
#         #     ),
#         #         row='all', col='all'
#         #     )
#
#         if trajectory:
#             trajectory = np.array(trajectory)
#             fig.add_trace(go.Scatter3d(
#                 # print(trajectory),
#                 x=trajectory[:, 0],
#                 y=trajectory[:, 1],
#                 z=-trajectory[:, 2],
#                 mode='markers+lines',
#                 marker=dict(
#                     size=5,
#                     color="black",
#                     showscale=False,
#                 ),
#                 line=dict(
#                     color="yellow",
#                     width=3,
#                     showscale=False,
#                 ),
#                 showlegend=False,
#             ),
#                 row='all', col='all'
#             )
#
#         fig.add_trace(go.Scatter3d(
#             x=[self.knowledge.current_location.x],
#             y=[self.knowledge.current_location.y],
#             z=[-self.knowledge.current_location.z],
#             mode='markers',
#             marker=dict(
#                 size=20,
#                 color="red",
#                 showscale=False,
#             ),
#             showlegend=False,  # remove all unnecessary trace names
#         ),
#             row='all', col='all'
#         )
#
#         camera = dict(
#             up=dict(x=0, y=0, z=1),
#             center=dict(x=0, y=0, z=0),
#             eye=dict(x=2.25, y=2.25, z=2.25)
#         )
#
#         fig.update_layout(
#             title={
#                 'text': "Simulation",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'},
#             scene=dict(
#                 zaxis=dict(nticks=4, range=[-3, 0], ),
#                 xaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 yaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 zaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 xaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
#                 yaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
#                 zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
#             ),
#             scene_aspectmode='manual',
#             scene_aspectratio=dict(x=1, y=1, z=.5),
#             scene2=dict(
#                 zaxis=dict(nticks=4, range=[-3, 0], ),
#                 xaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 yaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 zaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 xaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
#                 yaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
#                 zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
#             ),
#             scene2_aspectmode='manual',
#             scene2_aspectratio=dict(x=1, y=1, z=.5),
#             scene3=dict(
#                 zaxis=dict(nticks=4, range=[-3, 0], ),
#                 xaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 yaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 zaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 xaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
#                 yaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
#                 zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
#             ),
#             scene3_aspectmode='manual',
#             scene3_aspectratio=dict(x=1, y=1, z=.5),
#             scene_camera=camera,
#             scene2_camera=camera,
#             scene3_camera=camera,
#         )
#
#         # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
#         # if self.html:
#         # print("Save html")
#         plotly.offline.plot(fig, filename=self.filename + ".html", auto_open=False)
#         # os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html")
#         # fig.write_image(self.filename+".png", width=1980, height=1080, engine = "orca")






