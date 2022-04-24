#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import time
import os
from datetime import datetime
from scipy.stats import mvn, norm
from DataHandler import DataHandler
from AUV import AUV
import rospy
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
from usr_func import *


'''
The following program needs to be run before the adpative mission to prepare the prior and GP setup
'''

class MASCOT(AUV, DataHandler):
    ind_start, ind_now, ind_pre, ind_cand, ind_next = [0, 0, 0, 0, 0]  # only use index to make sure it is working properly
    mu_cond, Sigma_cond, F = [None, None, None]  # conditional mean and covariance and design matrix
    mu_prior, Sigma_prior = [None, None]  # prior mean and covariance matrix
    travelled_waypoints = None  # track how many waypoins have been explored
    data_path_waypoint = []  # save the waypoint to compare
    data_path_lat = []  # save the waypoint lat
    data_path_lon = []  # waypoint lon
    data_path_depth = []  # waypoint depth
    Total_waypoints = 60  # total number of waypoints to be explored
    counter_plot_simulation = 0  # track the plot, will be deleted
    distance_poly = 100  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2]  # [m], distance in depth, depth to be explored
    distanceTolerance = .1  # [m], distance tolerance for the neighbouring points
    distance_neighbours = np.sqrt(distance_poly ** 2 + (depth_obs[1] - depth_obs[0]) ** 2)

    def __init__(self):
        AUV.__init__(self)
        print("range of neighbours: ", self.distance_neighbours)
        self.load_global_path()
        self.load_prior()
        self.travelled_waypoints = 0
        self.get_starting_loc()
        self.run()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def load_prior(self):
        self.prior_extracted_path = self.path_global + "/Data/Corrected/Prior_extracted.txt"
        self.sigma_path = self.path_global + "/Config/Sigma_sal.txt"
        self.threshold_path = self.path_global + "/Config/threshold.txt"
        self.R_sal_path = self.path_global + "/Config/R_sal.txt"
        self.data_prior = np.loadtxt(self.prior_extracted_path, delimiter=", ")
        self.lat_loc = self.data_prior[:, 0]
        self.lon_loc = self.data_prior[:, 1]
        self.depth_loc = self.data_prior[:, 2]
        self.salinity_prior_corrected = self.data_prior[:, -1]
        self.mu_prior = self.salinity_prior_corrected
        self.Sigma_prior = np.loadtxt(self.sigma_path, delimiter=", ")
        self.Threshold_S = np.loadtxt(self.threshold_path, delimiter=", ")
        self.R_sal = np.loadtxt(self.R_sal_path, delimiter=", ")
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        self.N = len(self.mu_prior)
        self.F = np.zeros([1, len(self.mu_cond)])
        print("Prior for salinity is loaded correctly!!!")
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)
        print("mu_cond shape is: ", self.mu_cond.shape)
        print("Sigma_cond shape is: ", self.Sigma_cond.shape)
        print("Threshold: ", self.Threshold_S)
        print("R_sal: ", self.R_sal)
        print("N: ", self.N)

    def get_starting_loc(self):
        self.ind_start = int(np.loadtxt(self.path_global + "/Config/ind_start.txt", delimiter=", "))
        print("ind_start: ", self.ind_start)
        self.ind_next = self.ind_start
        self.updateF(self.ind_next)
        self.data_path_lat.append(self.lat_loc[self.ind_start])
        self.data_path_lon.append(self.lon_loc[self.ind_start])
        self.auv_handler.setWaypoint(deg2rad(self.lat_loc[self.ind_start]), deg2rad(self.lon_loc[self.ind_start]),
                                     -self.depth_loc[self.ind_start], speed=self.speed)
        self.updateWaypoint()

    def updateF(self, ind):
        self.F = np.zeros([1, self.N])
        self.F[0, ind] = True

    def EP_1D(self, mu, Sigma, Threshold):
        EP = np.zeros_like(mu)
        for i in range(EP.shape[0]):
            EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
        return EP

    def find_candidates_loc(self):
        '''
        find the candidates location based on distance coverage
        '''
        delta_x, delta_y = latlon2xy(self.lat_loc, self.lon_loc, self.lat_loc[self.ind_now],
                                          self.lon_loc[self.ind_now])  # using the distance
        delta_z = self.depth_loc - self.depth_loc[self.ind_now]  # depth distance in z-direction
        self.distance_vector = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        self.ind_cand = np.where(self.distance_vector <= self.distance_neighbours + self.distanceTolerance)[0]

    def GPupd(self, y_sampled):
        C = self.F @ self.Sigma_cond @ self.F.T + self.R_sal
        self.mu_cond = self.mu_cond + self.Sigma_cond @ self.F.T @ np.linalg.solve(C,(y_sampled - self.F @ self.mu_cond))
        self.Sigma_cond = self.Sigma_cond - self.Sigma_cond @ self.F.T @ np.linalg.solve(C, self.F @ self.Sigma_cond)

    def EIBV_1D(self, threshold, mu, Sig, F, R):
        Sigxi = Sig @ F.T @ np.linalg.solve(F @ Sig @ F.T + R, F @ Sig)
        V = Sig - Sigxi
        sa2 = np.diag(V).reshape(-1, 1)  # the corresponding variance term for each location
        IntA = 0.0
        for i in range(len(mu)):
            sn2 = sa2[i]
            m = mu[i]
            IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
        return IntA

    def get_normalised_distance_vector(self):
        dx, dy = latlon2xy(self.lat_loc, self.lon_loc, self.lat_loc[self.ind_now], self.lon_loc[self.ind_now])
        dz = self.depth_loc - self.depth_loc[self.ind_now]
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dist_normalised = dist / np.amax(dist)
        return dist_normalised

    def find_next_EIBV_1D(self, mu, Sig):
        id = []  # ind vector for containing the filtered desired candidate location
        t1 = time.time()
        dx1, dy1 = latlon2xy(self.lat_loc[self.ind_now], self.lon_loc[self.ind_now], self.lat_loc[self.ind_pre],
                                  self.lon_loc[self.ind_pre])
        dz1 = self.depth_loc[self.ind_now] - self.depth_loc[self.ind_pre]
        lat_cand_plot = []
        lon_cand_plot = []
        depth_cand_plot = []
        vec1 = np.array([dx1, dy1, dz1]).squeeze()
        for i in range(len(self.ind_cand)):
            if self.ind_cand[i] != self.ind_now:
                dx2, dy2 = latlon2xy(self.lat_loc[self.ind_cand[i]], self.lon_loc[self.ind_cand[i]],
                                          self.lat_loc[self.ind_now], self.lon_loc[self.ind_now])
                dz2 = self.depth_loc[self.ind_cand[i]] - self.depth_loc[self.ind_now]
                vec2 = np.array([dx2, dy2, dz2]).squeeze()
                if np.dot(vec1, vec2) > 0:
                    if dx2 == 0 and dy2 == 0:
                        print("Sorry, I cannot dive or float directly")
                        pass
                    else:
                        id.append(self.ind_cand[i])
                        lat_cand_plot.append(self.lat_loc[self.ind_cand[i]])
                        lon_cand_plot.append(self.lon_loc[self.ind_cand[i]])
                        depth_cand_plot.append(self.depth_loc[self.ind_cand[i]])
        id = np.unique(np.array(id))
        self.ind_cand = id
        M = len(id)
        eibv = []
        for k in range(M):
            F = np.zeros([1, self.N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(self.Threshold_S, mu, Sig, F, self.R_sal))
        t2 = time.time()

        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            print("Refinding new location")
            dist_normalised = self.get_normalised_distance_vector() # it moves to the location with the mean of both distance and ep
            self.ind_next = (np.abs(self.EP_1D(mu, Sig, self.Threshold_S) - .5) + dist_normalised).argmin()  # if not found next, use the other one
        else:
            print("Neighbouring location has next waypoint")
            self.ind_next = self.ind_cand[np.argmin(np.array(eibv))]

        self.data_path_lat.append(self.lat_loc[self.ind_next])
        self.data_path_lon.append(self.lon_loc[self.ind_next])
        self.data_path_depth.append(self.depth_loc[self.ind_next])
        print("Finding next waypoint takes: ", t2 - t1)
        self.updateF(self.ind_next)

    def updateWaypoint(self):
        # Since the accurate location of lat lon might have numerical problem for selecting the candidate location
        self.ind_pre = self.ind_now
        self.ind_now = self.ind_next

    def send_SMS_mission_complete(self):
        print("Mission complete Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        x_auv = self.vehicle_pos[0]
        y_auv = self.vehicle_pos[1]
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        SMS.contents.data = "Congrats, Adaptive Mission complete. LAUV-Xplore-1 location: " + str(lat_auv) + ", " + str(lon_auv)
        self.sms_pub_.publish(SMS)

    def save_mission_data(self):
        if np.around((self.t2 - self.t1), 0) % 10 == 0:
            print("Data is saved: ", self.counter_data_saved, " times")
            self.save_data()
            self.counter_data_saved = self.counter_data_saved + 1

    def append_mission_data(self):
        self.append_salinity(self.currentSalinity)
        self.append_temperature(self.currentTemperature)
        lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin,
                                                self.lon_origin)
        self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])
        self.append_timestamp(datetime.now().timestamp())
        self.append_ind_now(self.ind_now)
        self.append_sal_sampled(self.sal_sampled)

    def send_starting_waypoint(self):
        self.auv_handler.setWaypoint(deg2rad(self.lat_loc[self.ind_start]),deg2rad(self.lon_loc[self.ind_start]),-self.depth_loc[self.ind_start],speed = self.speed)
        print("moving to the starting location")

    def send_next_waypoint(self):
        self.auv_handler.setWaypoint(deg2rad(self.lat_loc[self.ind_next]),deg2rad(self.lon_loc[self.ind_next]),-self.depth_loc[self.ind_next],speed = self.speed)
        print("next waypoint", self.lat_loc[self.ind_next],self.lon_loc[self.ind_next],-self.depth_loc[self.ind_next])

    def run(self):
        self.createDataPath(self.path_global)
        self.t1 = time.time()
        # self.counter_waypoint = 0
        self.sal_sampled = 0
        self.counter_data_saved = 0
        self.send_starting_waypoint()

        while not rospy.is_shutdown():
            if self.init:
                self.t2 = time.time()
                self.append_mission_data()
                self.save_mission_data()
                print(self.auv_handler.getState())
                print("Counter waypoint: ", self.travelled_waypoints)
                print("Elapsed time after surfacing: ", self.t2 - self.t1)

                if (self.t2 - self.t1) / self.maxtime_underwater >= 1 and (
                        self.t2 - self.t1) % self.maxtime_underwater >= 0:
                    print("Longer than 10 mins, need a long break")
                    print(time.time())
                    self.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.popup_duration,
                                           phone_number=self.phone_number,
                                           iridium_dest=self.iridium_destination)  # self.ada_state = "surfacing"
                    print(time.time())
                    print("finished popup")
                    self.t1 = time.time()
                    self.t2 = time.time()

                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    self.sal_sampled = np.mean(self.data_salinity[-10:])  # take the past ten samples and average
                    print("Sampled salinity: ", self.sal_sampled)
                    self.GPupd(self.sal_sampled)  # update the field when it arrives the specified location
                    self.find_candidates_loc()
                    self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
                    self.updateWaypoint()
                    self.travelled_waypoints += 1
                    if self.travelled_waypoints >= self.Total_waypoints:
                        self.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.popup_duration,
                                               phone_number=self.phone_number,
                                               iridium_dest=self.iridium_destination)  # self.ada_state = "surfacing"
                        self.t1 = time.time()
                        self.t2 = time.time()
                        self.auv_handler.setWaypoint(deg2rad(self.lat_loc[self.ind_now]),
                                                     deg2rad(self.lon_loc[self.ind_now]), 0, speed=self.speed)
                        self.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    self.send_next_waypoint()

                print("vehicle state:", self.auv_handler.getState())
                print("pop duration: ", self.popup_duration)
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()


if __name__ == "__main__":
    a = MASCOT()
    print("Mission complete!!!")

