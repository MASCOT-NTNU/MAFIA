"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import os
import numpy as np
from usr_func import latlon2xy

# == GP kernel
THRESHOLD = 27
# ==

# ==
DISTANCE_SAME_LOCATION = 1
# ==

# == Grid
DISTANCE_NEIGHBOUR = 32
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

# == Boundary box
BOX = np.load(FILEPATH+"models/grid.npy")
LAT_BOX = BOX[:, 2]
LON_BOX = BOX[:, -1]
LATITUDE_ORIGIN = LAT_BOX[0]
LONGITUDE_ORIGIN = LON_BOX[0]
xbox, ybox = latlon2xy(LAT_BOX, LON_BOX, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
ROTATED_ANGLE = np.math.atan2(xbox[1] - xbox[0], ybox[1] - ybox[0])
# ==

# == PLotting
from matplotlib.cm import get_cmap
CMAP = get_cmap("BrBG", 10)
# ==
