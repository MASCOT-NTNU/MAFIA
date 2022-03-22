"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

"""
Usage:
next_location = MyopicPlanning3D(Knowledge, Experience).next_waypoint
"""

from usr_func import *
from MAFIA.Simulation.Field.Grid.Location import *
from MAFIA.Simulation.Kernel.Kernel import Kernel
import time


class MyopicPlanning3D:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.kernel = Kernel(self.knowledge)
        self.find_next_waypoint()

    def find_next_waypoint(self):
        self.filter_neighbouring_loc()
        t1 = time.time()
        eibv = []
        for k in range(len(self.knowledge.ind_neighbour_filtered)):
            # print("ind k: ", self.knowledge.ind_neighbour_filtered[k])
            eibv.append(self.kernel.get_eibv_1d(self.knowledge.ind_neighbour_filtered[k]))
        t2 = time.time()
        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            while True:
                self.knowledge.next_location = self.search_for_new_location()
                # TODO: add go home function
        else:
            self.knowledge.next_location = \
                self.get_location_from_ind(self.knowledge.ind_neighbour_filtered[np.argmin(np.array(eibv))])
        # print(eibv)
        print("Next location: ",
              self.knowledge.next_location.lat,
              self.knowledge.next_location.lon,
              self.knowledge.next_location.depth)
        print("Time consumed: ", t2 - t1)

    def filter_neighbouring_loc(self):
        t1 = time.time()
        id = []
        self.ind_neighbour_locations = self.knowledge.neighbour_hash_table[self.knowledge.ind_current_location[0]]
        # print("Before filtering: ", self.ind_neighbour_locations)
        vec1 = self.get_vector_between_locations(self.knowledge.previous_location, self.knowledge.current_location)
        for i in range(len(self.ind_neighbour_locations)):
            if self.ind_neighbour_locations[i] != self.knowledge.ind_current_location:
                vec2 = self.get_vector_between_locations(self.knowledge.current_location,
                                                         self.get_location_from_ind(self.ind_neighbour_locations[i]))
                if np.dot(vec1.T, vec2) >= 0:
                    id.append(self.ind_neighbour_locations[i])
        t2 = time.time()
        self.knowledge.ind_neighbour_filtered = np.unique(np.array(id))
        # print("Filtering takes: ", t2 - t1)
        # print("after filtering: ", self.knowledge.ind_neighbour_filtered)

    def get_vector_between_locations(self, loc_start, loc_end):
        dx, dy, dz = latlondepth2xyz(loc_end.lat, loc_end.lon, loc_end.depth,
                                     loc_start.lat, loc_start.lon, loc_start.depth)
        return vectorise([dx, dy, dz])

    def get_location_from_ind(self, ind):
        return Location(self.knowledge.coordinates[ind, 0],
                        self.knowledge.coordinates[ind, 1],
                        self.knowledge.coordinates[ind, 2])

    def search_for_new_location(self):
        ind_next = np.abs(get_excursion_prob_1d(self.knowledge.mu_cond, self.knowledge.Sigma_cond,
                                                self.knowledge.threshold) - .5).argmin()
        return self.get_location_from_ind(ind_next)

    # @property
    # def next_waypoint(self):
    #     return self.knowledge.coordinates[self.knowledge.ind_next, 0], \
    #            self.knowledge.coordinates[self.knowledge.ind_next, 1], \
    #            self.knowledge.coordinates[self.knowledge.ind_next, 2]





