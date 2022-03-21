"""
This script only contains the location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

import numpy as np
from usr_func import latlon2xy
from MAFIA.Simulation.Config.Config import *


class Location:

    def __init__(self, lat=None, lon=None, depth=None):
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.x, self.y = latlon2xy(self.lat, self.lon, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        self.z = self.depth


def get_distance_between_locations(loc1, loc2):
    dist_x, dist_y = latlon2xy(loc1.lat, loc1.lon, loc2.lat, loc2.lon)
    dist_z = np.abs(loc1.depth - loc2.depth)
    return np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

