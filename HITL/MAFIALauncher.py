"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""

from usr_func import *
from Config.Config import *
from Config.AdaframeConfig import *
from PlanningStrategies.Myopic3D import MyopicPlanning3D
from Knowledge.Knowledge import Knowledge
from AUV import AUV
from spde import spde
import pickle

# == Set up
LAT_START = 63.448747
LON_START = 10.416038
DEPTH_START = .5
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
Z_START = DEPTH_START
# ==


class MAFIALauncher:

    def __init__(self):
        self.load_waypoint()
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.load_prior()
        self.update_knowledge()
        self.load_hash_neighbours()
        self.load_hash_waypoint2gmrf()
        self.initialise_function_calls()
        self.setup_AUV()
        print("S1-S9 complete!")

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        print("S1: Waypoint is loaded successfully!")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
        print("S2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2)
        print("S3: GMRF model is loaded successfully!")

    def load_prior(self):
        print("S4: Prior is loaded successfully!")
        pass

    def update_knowledge(self):
        self.knowledge = Knowledge(gmrf_grid=self.gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
        print("S5: Knowledge of the field is set up successfully!")

    def load_hash_neighbours(self):
        neighbour_file = open(FILEPATH + "Config/HashNeighbours.p", 'rb')
        self.hash_neighbours = pickle.load(neighbour_file)
        neighbour_file.close()
        print("S6: Neighbour hash table is loaded successfully!")

    def load_hash_waypoint2gmrf(self):
        waypoint2gmrf_file = open(FILEPATH + "Config/HashWaypoint2GMRF.p", 'rb')
        self.hash_waypoint2gmrf = pickle.load(waypoint2gmrf_file)
        waypoint2gmrf_file.close()
        print("S7: Waypoint2GMRF hash table is loaded successfully!")

    def initialise_function_calls(self):
        get_ind_at_location3d_xyz(self.waypoints, 1, 2, 3)  # used to initialise the function call
        print("S8: Function calls are initialised successfully!")

    def setup_AUV(self):
        self.auv = AUV()
        print("S9: AUV is setup successfully!")

    def run(self):
        self.counter_waypoint = 0
        self.salinity = []
        ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, X_START, Y_START, Z_START)
        ind_previous_waypoint = ind_current_waypoint
        ind_visited_waypoint = []
        ind_visited_waypoint.append(ind_current_waypoint)
        while not rospy.is_shutdown():
            if self.auv.init:
                self.salinity.append(self.auv.currentSalinity)
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")

                    ind_sample_gmrf = self.hash_waypoint2gmrf[ind_current_waypoint]
                    self.salinity_measured = np.mean(self.salinity[-10:])

                    print("Sampled salinity: ", self.salinity_measured)

                    t1 = time.time()
                    self.gmrf_model.update(rel=self.salinity_measured, ks=ind_sample_gmrf)
                    t2 = time.time()
                    print("Update consumed: ", t2 - t1)

                    self.knowledge.mu = self.gmrf_model.mu
                    self.knowledge.SigmaDiag = self.gmrf_model.mvar()

                    self.pathplanner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints,
                                                        gmrf_model=self.gmrf_model, ind_current=ind_current_waypoint,
                                                        ind_previous=ind_previous_waypoint,
                                                        hash_neighbours=self.hash_neighbours,
                                                        hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                                        ind_visited=ind_visited_waypoint)

                    self.counter_waypoint += 1

                    ind_previous_waypoint = ind_current_waypoint
                    ind_current_waypoint = self.pathplanner.ind_next
                    ind_visited_waypoint.append(ind_current_waypoint)

                    x_waypoint = self.waypoints[ind_current_waypoint, 0]
                    y_waypoint = self.waypoints[ind_current_waypoint, 1]
                    z_waypoint = self.waypoints[ind_current_waypoint, 2]
                    lat_waypoint, lon_waypoint = xy2latlon(x_waypoint, y_waypoint, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)

                    if self.counter_waypoint >= NUM_STEPS:
                        self.auv_handler.setWaypoint(deg2rad(lat_waypoint),
                                                     deg2rad(lon_waypoint), 0, speed=self.auv.speed)
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    else:
                        self.auv_handler.setWaypoint(deg2rad(lat_waypoint),
                                                     deg2rad(lon_waypoint), z_waypoint, speed=self.auv.speed)
                        print("previous ind: ", ind_previous_waypoint)
                        print("current ind: ", ind_current_waypoint)

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

if __name__ == "__main__":
    s = MAFIALauncher()
    s.run()





