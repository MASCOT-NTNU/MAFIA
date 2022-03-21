"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from usr_func import *


class Knowledge:

    def __init__(self, coordinates=None, threshold=None, spde_model=None, previous_location=None, current_location=None,
                 distance_lateral=None, distance_vertical=None, distance_tolerance=None, distance_self=None):
        # knowing
        self.coordinates = coordinates
        self.excursion_prob = None
        self.excursion_set = None

        self.previous_location = previous_location
        self.current_location = current_location
        self.distance_lateral = distance_lateral
        self.distance_vertical = distance_vertical
        self.distance_tolerance = distance_tolerance
        self.distance_neighbours = np.sqrt(distance_lateral ** 2 + distance_vertical ** 2) + distance_tolerance
        self.distance_self = distance_self
        self.threshold = threshold
        self.kernel = spde_model

        # learned
        self.ind_cand = [] # save all potential candidate locations
        self.ind_cand_filtered = [] # save filtered candidate locations, [#1-No-PopUp-Dive, #2-No-Sharp-Turn]
        self.next_location = None
        self.ind_visited = []
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distance_travelled = [0]

