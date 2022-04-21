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
from MAFIA.Simulation.Field.Grid.Location import Location
import time


class MyopicPlanning3D:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.find_next_waypoint()

    def find_next_waypoint(self):
        self.filter_neighbouring_loc()
        t1 = time.time()
        eibv = []
        for k in range(len(self.knowledge.ind_neighbour_filtered_waypoint)):
            ind_candidate_waypoint = self.knowledge.ind_neighbour_filtered_waypoint[k]
            ind_candidate_grid = self.get_ind_candidate_from_waypoint_ind(ind_candidate_waypoint)
            eibv.append(self.get_eibv_1d(ind_candidate_grid))
        t2 = time.time()
        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            while True:
                pass
                break
                # self.knowledge.next_location = self.search_for_new_location()
                # TODO: add go home function
        else:
            ind_minimum_eibv = np.argmin(np.array(eibv))
            ind_next = self.knowledge.ind_neighbour_filtered_waypoint[ind_minimum_eibv]
            self.knowledge.next_location = self.get_location_from_ind_waypoint(ind_next)
        print("Time consumed: ", t2 - t1)

    def filter_neighbouring_loc(self):
        t1 = time.time()
        id = []
        self.ind_neighbour_locations = self.knowledge.neighbour_hash_table_waypoint[
            self.knowledge.ind_current_location_waypoint]
        vec1 = self.get_vector_between_locations(self.knowledge.previous_location, self.knowledge.current_location)
        for i in range(len(self.ind_neighbour_locations)):
            ind_candidate_location = self.ind_neighbour_locations[i]
            if ind_candidate_location != self.knowledge.ind_current_location_waypoint:
                vec2 = self.get_vector_between_locations(self.knowledge.current_location,
                                                         self.get_location_from_ind_waypoint(ind_candidate_location))
                if np.dot(vec1.T, vec2) >= 0:
                    id.append(self.ind_neighbour_locations[i])
        t2 = time.time()
        self.knowledge.ind_neighbour_filtered_waypoint = np.unique(np.array(id))
        # print("Filtering takes: ", t2 - t1)
        # print("after filtering: ", self.knowledge.ind_neighbour_filtered)

    def get_ind_candidate_from_waypoint_ind(self, ind_waypoint):
        location_candidate = self.get_location_from_ind_waypoint(ind_waypoint)
        ind_candidate_grid = get_ind_at_location3d_xyz(self.knowledge.coordinates_grid,
                                                       location_candidate.x,
                                                       location_candidate.y,
                                                       location_candidate.z)
        return ind_candidate_grid

    def get_vector_between_locations(self, loc_start, loc_end):
        dx = loc_end.x - loc_start.x
        dy = loc_end.y - loc_start.y
        dz = loc_end.z - loc_start.z
        return vectorise([dx, dy, dz])

    def get_location_from_ind_waypoint(self, ind):
        return Location(self.knowledge.coordinates_waypoint[ind, 0],
                        self.knowledge.coordinates_waypoint[ind, 1],
                        self.knowledge.coordinates_waypoint[ind, 2])

    def get_location_from_ind_grid(self, ind):
        return Location(self.knowledge.coordinates_grid[ind, 0],
                        self.knowledge.coordinates_grid[ind, 1],
                        self.knowledge.coordinates_grid[ind, 2])

    # def search_for_new_location(self):
    #     ind_next = np.abs(get_excursion_prob_1d(self.knowledge.mu_cond, self.knowledge.Sigma_cond,
    #                                             self.knowledge.threshold) - .5).argmin()
    #     return self.get_location_from_ind(ind_next)

    def get_eibv_1d(self, ind_candidate):
        variance_post = self.knowledge.spde_model.candidate(ks=ind_candidate) # update the field
        eibv = 0
        for i in range(self.knowledge.mu_cond.shape[0]):
            eibv += (mvn.mvnun(-np.inf, self.knowledge.threshold, self.knowledge.mu_cond[i], variance_post[i])[0] -
                     mvn.mvnun(-np.inf, self.knowledge.threshold, self.knowledge.mu_cond[i], variance_post[i])[0] ** 2)
        return eibv

    def get_excursion_prob_1d(self):
        Variance = self.knowledge.spde_model.mvar()
        excursion_prob = np.zeros_like(self.knowledge.mu_cond)
        for i in range(excursion_prob.shape[0]):
            excursion_prob[i] = norm.cdf(self.knowledge.threshold, self.knowledge.mu_cond[i], Variance[i, i])
        return excursion_prob

    def get_excursion_set(self):
        excursion_set = np.zeros_like(self.knowledge.mu_cond)
        excursion_set[self.knowledge.mu_cond < self.knowledge.threshold] = True
        return excursion_set




