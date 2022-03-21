"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import pandas as pd

from MAFIA.Simulation.Config.Config import *
from MAFIA.Simulation.Field.Knowledge.Knowledge import Knowledge
from MAFIA.Simulation.Field.Grid.Location import *
from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Kernel.Kernel import Kernel
from MAFIA.spde import spde
from MAFIA.Simulation.Plotting.plotting_func import *

FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/"


class PathPlanner:

    trajectory = []
    distance_travelled = 0
    waypoint_return_counter = 0

    def __init__(self, starting_location=None, goal_location=None):
        self.starting_location = starting_location
        self.goal_location = goal_location
        # self.gp.mu_cond = np.zeros_like(self.gp.mu_cond) # TODO: Wrong prior
        self.knowledge = Knowledge(coordinates=self.coordinates, threshold=THRESHOLD, spde_model=self.spde_model,
                                   )

    def setup_pathplanner(self):
        self.spde_model = spde(model=2)
        self.mu_prior = self.spde_model.mu
        self.coordinates = pd.read_csv(FILEPATH+"Simulation/Field/Grid/Grid.csv").to_numpy()

        pass

    def run(self):
        self.current_location = self.starting_location
        self.previous_location = self.current_location  #TODO: dot product will be zero, no effect on the first location.
        self.trajectory.append(self.current_location)



if __name__ == "__main__":
    starting_location = Location(63.455674, 10.429927, .5)
    goal_location = Location(63.440887, 10.354804, .5)
    p = PathPlanner(starting_location=starting_location, goal_location=goal_location)
    # p.run()




