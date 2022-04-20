"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import os


# == GP kernel
THRESHOLD = 27
# ==

# ==
DISTANCE_SAME_LOCATION = 1
# ==

# == Grid
DISTANCE_NEIGHBOUR = 32
LATITUDE_ORIGIN = 0
LONGITUDE_ORIGIN = 0
DEPTH_ORIGIN = 0
DISTANCE_LATERAL = 32
DISTANCE_VERTICAL = .5
DISTANCE_SELF = 5
# ==

# == Path planner
NUM_STEPS = 80
working_directory = os.getcwd()
FIGPATH = working_directory + "/MAFIA/fig/"
FILEPATH = working_directory + "/MAFIA/"
# ==

# == PLotting
from matplotlib.cm import get_cmap
CMAP = get_cmap("BrBG", 10)
# ==


