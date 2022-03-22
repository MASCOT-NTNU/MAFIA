"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from usr_func import get_ind_at_location


class Knowledge:

    def __init__(self, coordinates=None, neighbour_hash_table=None, threshold=None, spde_model=None,
                 previous_location=None, current_location=None):
        # knowing
        self.coordinates = coordinates
        self.excursion_prob = None
        self.excursion_set = None

        self.previous_location = previous_location
        self.current_location = current_location
        self.ind_previous_location = get_ind_at_location(self.coordinates, self.previous_location)
        self.ind_current_location = get_ind_at_location(self.coordinates, self.current_location)
        self.neighbour_hash_table = neighbour_hash_table
        self.threshold = threshold
        self.spde_model = spde_model

        # learned
        self.ind_neighbour_filtered = []
        self.next_location = None
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distance_travelled = [0]

