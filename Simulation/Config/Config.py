"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""
import os
import numpy as np
from usr_func import latlon2xy

# == Sys
working_directory = os.getcwd()
FIGPATH = working_directory + "/MAFIA/fig/"
FILEPATH = working_directory + "/MAFIA/"
# ==

# == GP kernel
THRESHOLD = 27
# ==

# == Path planner
NUM_STEPS = 80
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


