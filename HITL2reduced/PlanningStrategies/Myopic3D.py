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
from Config.Config import *
import time


vectorize(['float32(float32, float32)'], target='cuda')
def get_eibv_from_gpu(mu, SigmaDiag, threshold=THRESHOLD):
  cdf = norm.cdf(threshold, mu, SigmaDiag)
  bv = cdf*(1-cdf)
  ibv = np.sum(bv)
  return ibv


def get_eibv_from_fast(mu, sigma, threshold=THRESHOLD):
  p = norm.cdf(threshold, mu, sigma)
  bv = p * (1 - p)
  ibv = np.sum(bv)
  return ibv


class MyopicPlanning3D:

    def __init__(self, knowledge=None, waypoints=None, gmrf_model=None, ind_current=None, ind_previous=None,
                 hash_neighbours=None, hash_waypoint2gmrf=None, ind_visited=None):

        self.knowledge = knowledge
        self.waypoints = waypoints
        self.gmrf_model = gmrf_model
        self.ind_current = ind_current
        self.ind_previous = ind_previous
        self.hash_neighbours = hash_neighbours
        self.hash_waypoint2gmrf = hash_waypoint2gmrf
        self.ind_visited = ind_visited

        self.find_all_neighbours()
        self.smooth_filter_neighbours()
        self.find_next_waypoint_using_min_eibv()

    def find_all_neighbours(self):
        self.ind_neighbours = self.hash_neighbours[self.ind_current]

    def smooth_filter_neighbours(self):
        vec1 = self.get_vec_from_indices(self.ind_previous, self.ind_current)
        self.ind_candidates = []
        for i in range(len(self.ind_neighbours)):
            ind_candidate = self.ind_neighbours[i]
            if not ind_candidate in self.ind_visited:
                vec2 = self.get_vec_from_indices(self.ind_current, ind_candidate)
                if np.dot(vec1.T, vec2) >= 0:
                    self.ind_candidates.append(ind_candidate)

    def get_vec_from_indices(self, ind_start, ind_end):
        x_start = self.waypoints[ind_start, 0]
        y_start = self.waypoints[ind_start, 1]
        z_start = self.waypoints[ind_start, 2]

        x_end = self.waypoints[ind_end, 0]
        y_end = self.waypoints[ind_end, 1]
        z_end = self.waypoints[ind_end, 2]

        dx = x_end - x_start
        dy = y_end - y_start
        dz = z_end - z_start

        return vectorise([dx, dy, dz])

    def find_next_waypoint_using_min_eibv(self):
        self.EIBV = []
        t1 = time.time()
        for ind_candidate in self.ind_candidates:
            self.EIBV.append(self.get_eibv_from_gmrf_model(self.hash_waypoint2gmrf[ind_candidate]))
        if self.EIBV:
            self.ind_next = self.ind_candidates[np.argmin(self.EIBV)]
        else:
            self.ind_next = self.ind_neighbours[np.random.randint(len(self.ind_neighbours))]
        t2 = time.time()
        print("Path planning takes: ", t2 - t1)

    def get_eibv_from_gmrf_model(self, ind_candidate):
        t1 = time.time()
        variance_post = self.gmrf_model.candidate(ks=ind_candidate)  # update the field
        t2 = time.time()
        # print("Update variance take: ", t2 - t1)

        # eibv = get_eibv_from_gpu(self.knowledge.mu, variance_post)
        t1 = time.time()
        eibv = get_eibv_from_fast(self.knowledge.mu, variance_post, self.gmrf_model.threshold)
        t2 = time.time()
        # print("EIBV calculation takes: ", t2 - t1)
        return eibv



