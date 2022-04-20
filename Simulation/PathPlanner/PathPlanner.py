"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import pandas as pd

from usr_func import *

from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Kernel.Kernel import Kernel
from MAFIA.Simulation.Simulator.Sampler import Sampler
from MAFIA.Simulation.Field.Grid.Location import *
from MAFIA.Simulation.Field.Knowledge.Knowledge import Knowledge
from MAFIA.Simulation.Plotting.KnowledgePlot import KnowledgePlot
from MAFIA.spde import spde
import pickle

FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/"


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
        neighbour_hash_table_filehandler = open(FILEPATH + "Simulation/Field/Grid/Neighbours.p", 'rb')
        self.neighbour_hash_table = pickle.load(neighbour_hash_table_filehandler)
        neighbour_hash_table_filehandler.close()
        self.num_steps = NUM_STEPS
        self.current_location = self.starting_location
        self.previous_location = self.current_location
        self.knowledge = Knowledge(coordinates=self.coordinates, neighbour_hash_table=self.neighbour_hash_table,
                                   threshold=THRESHOLD, spde_model=self.spde_model, previous_location=self.previous_location,
                                   current_location=self.current_location)
        self.load_ground_truth()

    def load_ground_truth(self):
        path_mu_truth = FILEPATH + "Simulation/Field/Data/data_mu_truth.csv"
        self.ground_truth = pd.read_csv(path_mu_truth).to_numpy()

    def run(self):
        self.ind_sample = get_ind_at_location3d_wgs(self.coordinates, self.starting_location)
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
    starting_location = Location(63.45121, 10.40673, .5)
    p = PathPlanner(starting_location=starting_location)
    p.run()


#%%
path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/models/depth.npy'

df = np.load(path)
df.shape


