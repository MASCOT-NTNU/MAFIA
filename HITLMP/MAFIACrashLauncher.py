"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""

from usr_func import *
from Config.AUVConfig import * # !!!! ROSPY important
from Config.Config import *
from PlanningStrategies.Myopic3D import MyopicPlanning3D
from Knowledge.Knowledge import Knowledge
from AUV import AUV
from spde import spde
import pickle
import concurrent.futures

# == Set up
# LAT_START = 63.447231
# LON_START = 10.412948
# DEPTH_START = .5
# X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
# Z_START = DEPTH_START
VERTICES_TRANSECT = np.array([[63.450421, 10.395289],
                              [63.453768, 10.420457],
                              [63.446442, 10.412006]])
# ==


class MAFIA2Launcher:

    def __init__(self):
        self.load_waypoint()
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.update_knowledge()
        self.load_hash_neighbours()
        self.load_hash_waypoint2gmrf()
        self.initialise_function_calls()
        self.setup_AUV()
        self.update_time = rospy.get_time()
        self.setup_myopic3d_planner()
        print("S1-S10 complete!")

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        print("S1: Waypoint is loaded successfully!")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
        self.N_gmrf_grid = len(self.gmrf_grid)
        print("S2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        mu = np.load("models/mu2cond.npy")
        self.gmrf_model = spde(model=2, reduce=True, method=2)
        self.gmrf_model.mu = mu
        print("S3: GMRF model is loaded successfully! Mu is loaded successfully!")

    def update_knowledge(self):
        self.knowledge = Knowledge(gmrf_grid=self.gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
        print("S4: Knowledge of the field is set up successfully!")

    def load_hash_neighbours(self):
        neighbour_file = open(FILEPATH + "Config/HashNeighbours.p", 'rb')
        self.hash_neighbours = pickle.load(neighbour_file)
        neighbour_file.close()
        print("S5: Neighbour hash table is loaded successfully!")

    def load_hash_waypoint2gmrf(self):
        waypoint2gmrf_file = open(FILEPATH + "Config/HashWaypoint2GMRF.p", 'rb')
        self.hash_waypoint2gmrf = pickle.load(waypoint2gmrf_file)
        waypoint2gmrf_file.close()
        print("S6: Waypoint2GMRF hash table is loaded successfully!")

    def initialise_function_calls(self):
        get_ind_at_location3d_xyz(self.waypoints, 1, 2, 3)  # used to initialise the function call
        print("S7: Function calls are initialised successfully!")

    def setup_AUV(self):
        self.auv = AUV()
        print("S8: AUV is setup successfully!")

    def setup_myopic3d_planner(self):
        self.myopic3d_planner = MyopicPlanning3D(waypoints=self.waypoints, hash_neighbours=self.hash_neighbours,
                                                 hash_waypoint2gmrf=self.hash_waypoint2gmrf)
        print("S10: Myopic3D planner is setup successfully!")

    def run(self):
        self.counter_waypoint_adaptive = 0
        self.auv_data = []
        self.ind_visited_waypoint = []

        lat_waypoint, lon_waypoint = VERTICES_TRANSECT[-1, :]
        depth_waypoint = DEPTH_TOP

        x_start, y_start = latlon2xy(lat_waypoint, lon_waypoint, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        z_start = depth_waypoint

        self.ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, x_start, y_start, z_start)
        self.ind_previous_waypoint = self.ind_current_waypoint
        self.ind_visited_waypoint.append(self.ind_current_waypoint)
        self.set_waypoint_using_ind_waypoint(self.ind_current_waypoint)
        print("Start 2-step planning")
        self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            print("Start concurrent")
            executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
                            self.ind_current_waypoint,
                            self.ind_previous_waypoint,
                            self.ind_visited_waypoint)
            print("End concurrent")
        self.ind_next_waypoint = int(np.loadtxt(FILEPATH + "Waypoint/ind_next.txt"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            print("Start concurrent")
            executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
                            self.ind_next_waypoint,
                            self.ind_current_waypoint,
                            self.ind_visited_waypoint)
            print("End concurrent")
        self.ind_pioneer_waypoint = int(np.loadtxt(FILEPATH + "Waypoint/ind_next.txt"))
        print("Finished 2-step planning!!!")

        t_start = time.time()
        while not rospy.is_shutdown():
            if self.auv.init:
                print("Adaptive waypoint step: ", self.counter_waypoint_adaptive, " of , ", NUM_STEPS)
                t_end = time.time()
                self.auv_data.append([self.auv.vehicle_pos[0],
                                      self.auv.vehicle_pos[1],
                                      self.auv.vehicle_pos[2],
                                      self.auv.currentSalinity])
                self.auv.current_state = self.auv.auv_handler.getState()
                if ((t_end - t_start) / self.auv.max_submerged_time >= 1 and
                        (t_end - t_start) % self.auv.max_submerged_time >= 0):
                    print("Longer than 10 mins, need a long break")
                    self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                           phone_number=self.auv.phone_number,
                                           iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                    t_start = time.time()
                    self.popup = True

                if not self.popup:
                    # if self.auv.auv_handler.getState() == "waiting":
                    if (self.auv.auv_handler.getState() == "waiting" and
                            rospy.get_time() - self.update_time > WAYPOINT_UPDATE_TIME):
                        print("Arrived the current location")
                        self.ind_previous_waypoint = self.ind_current_waypoint
                        self.ind_current_waypoint = self.ind_next_waypoint
                        self.ind_next_waypoint = self.ind_pioneer_waypoint
                        self.ind_visited_waypoint.append(self.ind_current_waypoint)

                        x_waypoint = self.waypoints[self.ind_current_waypoint, 0]
                        y_waypoint = self.waypoints[self.ind_current_waypoint, 1]
                        lat_waypoint, lon_waypoint = xy2latlon(x_waypoint, y_waypoint, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                        depth_waypoint = self.waypoints[self.ind_current_waypoint, 2]

                        if self.counter_waypoint_adaptive >= NUM_STEPS:
                            self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                                   phone_number=self.auv.phone_number,
                                                   iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                            self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), 0,
                                                             speed=self.auv.speed)
                            print("Mission complete! Congrates!")
                            rospy.signal_shutdown("Mission completed!!!")
                            break
                        else:
                            self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), depth_waypoint,
                                                             speed=self.auv.speed)
                            print("Set waypoint successfully!")
                            self.update_time = rospy.get_time()

                        ind_assimilated, salinity_assimilated = self.assimilate_data(np.array(self.auv_data))
                        t1 = time.time()
                        self.gmrf_model.update(rel=salinity_assimilated, ks=ind_assimilated)
                        t2 = time.time()
                        print("Update consumed: ", t2 - t1)

                        self.knowledge.mu = self.gmrf_model.mu
                        self.knowledge.SigmaDiag = self.gmrf_model.mvar()

                        self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model)
                        # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            print("Start concurrent")
                            executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
                                            self.ind_next_waypoint,
                                            self.ind_current_waypoint,
                                            self.ind_visited_waypoint)
                            print("End concurrent")
                        self.ind_pioneer_waypoint = int(np.loadtxt(FILEPATH + "Waypoint/ind_next.txt"))
                        self.counter_waypoint_adaptive += 1
                else:
                    # if self.auv.auv_handler.getState() == "waiting":
                    if (self.auv.auv_handler.getState() == "waiting" and
                            rospy.get_time() - self.update_time > WAYPOINT_UPDATE_TIME):
                        self.popup = False

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
        print("dataset before filtering: ", dataset[-10:, :])
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= MIN_DEPTH_FOR_DATA_ASSIMILATION)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        print("dataset after filtering: ", dataset[-10:, :])
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
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return ind_assimilated, vectorise(salinity_assimilated)


if __name__ == "__main__":
    s = MAFIA2Launcher()
    s.run()


