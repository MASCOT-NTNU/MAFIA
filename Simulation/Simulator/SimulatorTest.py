"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""

from usr_func import *
from MAFIA.Simulation.Config.Config import *
from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Simulator.Sampler import Sampler
from MAFIA.Simulation.Knowledge.Knowledge import Knowledge
from MAFIA.spde import spde
import pickle

# == Set up
LAT_START = 63.453222
LON_START = 10.414687
DEPTH_START = 1.5
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
        ind_waypoint = get_ind_at_location3d_xyz(self.waypoints, X_START, Y_START, Z_START)
        ind_previous = ind_waypoint
        ind_visited = []
        ind_visited.append(ind_waypoint)
        for i in range(NUM_STEPS):
            print("Step: ", i)
            ind_sample = self.hash_waypoint2gmrf[ind_waypoint]
            self.salinity_measured = self.simulated_truth[ind_sample][0]

            t1 = time.time()
            self.gmrf_model.update(rel=self.salinity_measured, ks=ind_sample)
            t2 = time.time()
            print("Update consumed: ", t2 - t1)

            self.knowledge.mu = self.gmrf_model.mu
            self.knowledge.SigmaDiag = self.gmrf_model.mvar()

            planner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints, gmrf_model=self.gmrf_model,
                                       ind_current=ind_waypoint, ind_previous=ind_previous,
                                       hash_neighbours=self.hash_neighbours, hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                       ind_visited=ind_visited)
            ind_previous = ind_waypoint
            ind_waypoint = planner.ind_next
            ind_visited.append(ind_waypoint)
            print("previous ind: ", ind_previous)
            print("current ind: ", ind_waypoint)

        pass

if __name__ == "__main__":
    s = Simulator()
    s.run()

