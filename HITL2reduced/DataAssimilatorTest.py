"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""

from usr_func import *
from Config.AdaframeConfig import * # !!!! ROSPY important
from Config.Config import *
from PlanningStrategies.Myopic3D import MyopicPlanning3D
from Knowledge.Knowledge import Knowledge
from AUV import AUV
from spde import spde
import pickle

# == Set up
LAT_START = 63.447231
LON_START = 10.412948
DEPTH_START = .5
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
Z_START = DEPTH_START
# ==


class MAFIA2Launcher:

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
        self.update_time = rospy.get_time()
        print("S1-S9 complete!")

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        print("S1: Waypoint is loaded successfully!")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
        self.N_gmrf_grid = len(self.gmrf_grid)
        print("S2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2, reduce=True)
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
        self.auv_data = []

        ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, X_START, Y_START, Z_START)
        ind_previous_waypoint = ind_current_waypoint
        ind_visited_waypoint = []
        ind_visited_waypoint.append(ind_current_waypoint)

        self.set_waypoint_using_ind_waypoint(ind_current_waypoint)

        print("Start 2-step planning")
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
        print("Finished 2-step planning!!!")

        t_start = time.time()
        while not rospy.is_shutdown():
            if self.auv.init:
                print("Waypoint step: ", self.counter_waypoint)
                t_end = time.time()

                self.auv_data.append([self.auv.vehicle_pos[0],
                                      self.auv.vehicle_pos[1],
                                      self.auv.vehicle_pos[2],
                                      self.auv.currentSalinity])
                print('Appended data: ', self.auv.vehicle_pos[0], self.auv.vehicle_pos[1],
                      self.auv.vehicle_pos[2], self.auv.currentSalinity)
                self.auv.current_state = self.auv.auv_handler.getState()
                print("AUV state: ", self.auv.current_state)

                if ((t_end - t_start) / self.auv.max_submerged_time >= 1 and
                        (t_end - t_start) % self.auv.max_submerged_time >= 0):
                    print("Longer than 10 mins, need a long break")
                    self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                           phone_number=self.auv.phone_number,
                                           iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                    t_start = time.time()

                if self.auv.auv_handler.getState() == "waiting" and rospy.get_time() - self.update_time > WAYPOINT_UPDATE_TIME:
                    print("Arrived the current location")

                    ind_previous_waypoint = ind_current_waypoint
                    ind_current_waypoint = ind_next_waypoint
                    ind_next_waypoint = ind_pioneer_waypoint
                    ind_visited_waypoint.append(ind_current_waypoint)

                    x_waypoint = self.waypoints[ind_current_waypoint, 0]
                    y_waypoint = self.waypoints[ind_current_waypoint, 1]
                    z_waypoint = self.waypoints[ind_current_waypoint, 2]
                    lat_waypoint, lon_waypoint = xy2latlon(x_waypoint, y_waypoint, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)

                    if self.counter_waypoint >= NUM_STEPS:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                               phone_number=self.auv.phone_number,
                                               iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), 0,
                                                         speed=self.auv.speed)
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    else:
                        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), z_waypoint,
                                                         speed=self.auv.speed)
                        print("Set waypoint successfully!")
                        self.update_time = rospy.get_time()
                        print("previous ind: ", ind_previous_waypoint)
                        print("current ind: ", ind_current_waypoint)

                    ind_assimilated, salinity_assimilated = self.assimilate_data(np.array(self.auv_data))
                    print("Path mean salinity: ", np.mean(salinity_assimilated))

                    t1 = time.time()
                    # self.gmrf_model.update(rel=salinity_assimilated[0], ks=ind_assimilated[0]) #TODO: Avoid bug, for testing
                    self.gmrf_model.update(rel=vectorise(salinity_assimilated), ks=ind_assimilated)
                    t2 = time.time()
                    print("Update consumed: ", t2 - t1)

                    self.knowledge.mu = self.gmrf_model.mu
                    self.knowledge.SigmaDiag = self.gmrf_model.mvar()

                    self.pathplanner = MyopicPlanning3D(knowledge=self.knowledge, waypoints=self.waypoints,
                                                        gmrf_model=self.gmrf_model,
                                                        ind_current=ind_next_waypoint,
                                                        ind_previous=ind_current_waypoint,
                                                        hash_neighbours=self.hash_neighbours,
                                                        hash_waypoint2gmrf=self.hash_waypoint2gmrf,
                                                        ind_visited=ind_visited_waypoint)
                    ind_pioneer_waypoint = self.pathplanner.ind_next
                    self.counter_waypoint += 1

                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()

    def set_waypoint_using_ind_waypoint(self, ind_waypoint):
        x_waypoint = self.waypoints[ind_waypoint, 0]
        y_waypoint = self.waypoints[ind_waypoint, 1]
        z_waypoint = self.waypoints[ind_waypoint, 2]
        lat_waypoint, lon_waypoint = xy2latlon(x_waypoint, y_waypoint, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), z_waypoint, speed=self.auv.speed)
        print("Set waypoint successfully!")

    def assimilate_data(self, dataset):
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
        print("Ind assimilated: ", ind_assimilated)
        print("Salinity assimilated: ", np.mean(salinity_assimilated))
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        return ind_assimilated, salinity_assimilated


if __name__ == "__main__":
    s = MAFIA2Launcher()
    s.run()
