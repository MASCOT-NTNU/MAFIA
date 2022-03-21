"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from usr_func import *


class Knowledge:

    def __init__(self, coordinates=None, polygon_border=None, mu_prior=None, Sigma_prior=None, mu_cond=None,
                 Sigma_cond=None, threshold=None, kernel=None, ind_prev=None, ind_now=None, distance_lateral=None,
                 distance_vertical=None, distance_tolerance=None, distance_self=None):
        # knowing
        self.coordinates = coordinates
        self.polygon = polygon_border
        self.mu_prior = mu_prior
        self.Sigma_prior = Sigma_prior
        self.mu_cond = mu_cond
        self.Sigma_cond = Sigma_cond
        self.excursion_prob = None
        self.excursion_set = None

        self.ind_prev = ind_prev
        self.ind_now = ind_now
        self.distance_lateral = distance_lateral
        self.distance_vertical = distance_vertical
        self.distance_tolerance = distance_tolerance
        self.distance_neighbours = np.sqrt(distance_lateral ** 2 + distance_vertical ** 2) + distance_tolerance
        self.distance_self = distance_self
        self.threshold = threshold
        self.kernel = kernel

        # learned
        self.ind_cand = [] # save all potential candidate locations
        self.ind_cand_filtered = [] # save filtered candidate locations, [#1-No-PopUp-Dive, #2-No-Sharp-Turn]
        self.ind_next = []
        self.ind_visited = []
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distance_travelled = [0]

