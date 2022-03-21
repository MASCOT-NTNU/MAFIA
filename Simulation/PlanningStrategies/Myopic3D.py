"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

"""
Usage:
lat_next, lon_next = MyopicPlanning_2D(Knowledge, Experience).next_waypoint
"""

from usr_func import *
from MAFIA.Simulation.Field.Grid.Location import Location
import time


class MyopicPlanning3D:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.find_next_waypoint()

    def find_next_waypoint(self):
        self.find_candidates_loc()
        self.filter_candidates_loc()
        t1 = time.time()
        id = self.knowledge.ind_cand_filtered
        eibv = []
        for k in range(len(id)):
            F = getFVector(id[k], self.knowledge.coordinates.shape[0])
            eibv.append(get_eibv_1d(self.knowledge.threshold, self.knowledge.mu_cond,
                                    self.knowledge.Sigma_cond, F, self.knowledge.kernel.R))
        t2 = time.time()
        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            while True:
                ind_next = self.search_for_new_location()
                if not ind_next in self.knowledge.ind_visited:
                    self.knowledge.ind_next = ind_next
                    break
        else:
            self.knowledge.ind_next = self.knowledge.ind_cand_filtered[np.argmin(np.array(eibv))]

    def find_candidates_loc(self):
        delta_x, delta_y = latlon2xy(self.knowledge.coordinates[:, 0], self.knowledge.coordinates[:, 1],
                                     self.knowledge.current_location.lat, self.knowledge.current_location.lon)
        delta_z = self.knowledge.coordinates[:, 2] - self.knowledge.current_location.depth

        self.distance_euclidean = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        self.distance_ellipsoid = (delta_x ** 2 / (1.5 * self.knowledge.distance_lateral) ** 2) + \
                                  (delta_y ** 2 / (1.5 * self.knowledge.distance_lateral) ** 2) + \
                                  (delta_z ** 2 / (self.knowledge.distance_vertical + 0.3) ** 2)
        self.knowledge.ind_cand = np.where((self.distance_ellipsoid <= 1) *
                                           (self.distance_euclidean > self.knowledge.distance_self))[0]
        # print("ind:", self.knowledge.ind_cand)
        self.knowledge.candidate_locations = self.get_locations_from_indice(self.knowledge.ind_cand)

    def get_locations_from_indice(self, ind):
        return Location(self.knowledge.coordinates[ind, 0],
                        self.knowledge.coordinates[ind, 1],
                        self.knowledge.coordinates[ind, 2])

    def filter_candidates_loc(self):
        id = []  # ind vector for containing the filtered desired candidate location
        t1 = time.time()

        dz1 = (self.knowledge.current_location.depth - self.knowledge.previous_location.depth)
        vec1 = vectorise([dx1, dy1, dz1])

        for i in range(len(self.knowledge.ind_cand)):
            if self.knowledge.ind_cand[i] != self.knowledge.ind_now:
                if not self.knowledge.ind_cand[i] in self.knowledge.ind_visited:
                    dx2, dy2 = latlon2xy(self.knowledge.coordinates[self.knowledge.ind_cand[i], 0],
                                         self.knowledge.coordinates[self.knowledge.ind_cand[i], 1],
                                         self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                         self.knowledge.coordinates[self.knowledge.ind_now, 1])
                    dz2 = (self.knowledge.coordinates[self.knowledge.ind_cand[i], 2] -
                           self.knowledge.coordinates[self.knowledge.ind_now, 2])
                    vec2 = vectorise([dx2, dy2, dz2])
                    if np.dot(vec1.T, vec2) >= 0:
                        if dx2 == 0 and dy2 == 0:
                            pass
                        else:
                            id.append(self.knowledge.ind_cand[i])
        id = np.unique(np.array(id))  # filter out repetitive candidate locations
        self.knowledge.ind_cand_filtered = id  # refresh old candidate location
        t2 = time.time()
        # print("Filtering takes: ", t2 - t1)

    def get_vector_between_locations(self, loc_start, loc_end):
        dx, dy, dz = latlondepth2xyz(self.knowledge.current_location.lat,
                                     self.knowledge.current_location.lon,
                                        self.knowledge.current_location.depth,
                                        self.knowledge.previous_location.lat, self.knowledge.previous_location.lon)

    def search_for_new_location(self):
        ind_next = np.abs(get_excursion_prob_1d(self.knowledge.mu_cond, self.knowledge.Sigma_cond,
                                                self.knowledge.threshold) - .5).argmin()
        return ind_next

    @property
    def next_waypoint(self):
        return self.knowledge.coordinates[self.knowledge.ind_next, 0], \
               self.knowledge.coordinates[self.knowledge.ind_next, 1], \
               self.knowledge.coordinates[self.knowledge.ind_next, 2]



