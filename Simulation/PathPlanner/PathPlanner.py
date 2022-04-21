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
#%%

class PathPlanner:

    trajectory = []
    distance_travelled = 0
    waypoint_return_counter = 0

    def __init__(self, starting_location=None):
        self.starting_location = starting_location
        self.setup_pathplanner()

    def setup_pathplanner(self):
        self.spde_model = spde(model=2)
        self.coordinates = pd.read_csv(FILEPATH + "Simulation/Field/Grid/Grid.csv").to_numpy()
        self.coordinates_xyz = self.coordinates[:, -3:]
        neighbour_hash_table_filehandler = open(FILEPATH + "Simulation/Field/Grid/Neighbours.p", 'rb')
        self.neighbour_hash_table = pickle.load(neighbour_hash_table_filehandler)
        neighbour_hash_table_filehandler.close()
        self.num_steps = NUM_STEPS

        self.current_location = self.starting_location
        self.previous_location = self.current_location

        self.knowledge = Knowledge(coordinates=self.coordinates_xyz, neighbour_hash_table=self.neighbour_hash_table,
                                   threshold=THRESHOLD, spde_model=self.spde_model,
                                   previous_location=self.previous_location, current_location=self.current_location)
        self.load_ground_truth()

        self.initialise_function_calls()

    def initialise_function_calls(self):
        get_ind_at_location3d_xyz(self.coordinates_xyz, 1, 2, 3)  # used to initialise the function call

    def load_ground_truth(self):
        path_mu_truth = FILEPATH + "Simulation/Field/Data/data_mu_truth.csv"
        self.ground_truth = pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1)

    def run(self):
        self.ind_sample = get_ind_at_location3d_xyz(self.coordinates_xyz, self.starting_location.x,
                                                    self.starting_location.y, self.starting_location.z)
        for i in range(self.num_steps):
            print("Step No. ", i)
            self.trajectory.append(self.knowledge.current_location)
            Sampler(self.knowledge, self.ground_truth, self.ind_sample)
            myopic3d = MyopicPlanning3D(knowledge=self.knowledge)
            self.next_location = self.knowledge.next_location
            self.ind_sample = get_ind_at_location3d_wgs(self.coordinates, self.next_location)
            self.knowledge.step_no = i
            KnowledgePlot(knowledge=self.knowledge, vmin=0, vmax=32,
                          filename=FILEPATH + "fig/myopic3d/P_{:03d}".format(i), html=False)
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
sal = np.load(FILEPATH + "models/prior.npy")
lats = np.load(FILEPATH + "models/lats.npy")
lons = np.load(FILEPATH + "models/lons.npy")
depth = np.load(FILEPATH + "models/depth.npy")
grid = np.load(FILEPATH + "models/grid.npy")
print(os.listdir(FILEPATH+"models/"))
l = sal.shape[0]
lats = lats[:l]
lons = lons[:l]
depth = depth[:l]
#%%

#%%

import plotly.graph_objects as go
import numpy as np
import plotly
# Helix equation

fig = go.Figure(data=[go.Scatter3d(
    x=lons,
    y=lats,
    z=depth,
    mode='markers',
    marker=dict(
        size=12,
        color=sal,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8,
        cmin=0,
        cmax=30,
    )

)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename=FIGPATH+"prior.html", auto_open=True)


