"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from usr_func import get_ind_at_location3d_wgs


class Knowledge:

    def __init__(self, coordinates_grid=None, coordinates_waypoint=None, neighbour_hash_table_waypoint=None, threshold=None,
                 spde_model=None, previous_location=None, current_location=None):
        # known
        self.coordinates_grid = coordinates_grid
        self.coordinates_waypoint = coordinates_waypoint
        self.neighbour_hash_table_waypoint = neighbour_hash_table_waypoint
        self.threshold = threshold
        self.spde_model = spde_model
        self.mu_cond = self.spde_model.mu
        self.Sigma_cond_diag = self.spde_model.mvar()
        self.previous_location = previous_location
        self.current_location = current_location

        # computed
        self.ind_previous_location_waypoint = None
        self.ind_current_location_waypoint = None
        self.excursion_prob = None
        self.excursion_set = None

        # learned
        self.ind_neighbour_filtered_waypoint = []
        self.next_location = None
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distance_travelled = [0]

