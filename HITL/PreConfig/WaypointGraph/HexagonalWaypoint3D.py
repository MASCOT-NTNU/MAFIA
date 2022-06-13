"""
This script generates regular hexgonal grid points within certain boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-25
"""


# from usr_func import *
import time, numpy as np
from PreConfig.WaypointGraph.HexagonalWaypoint2D import HexgonalGrid2DGenerator


class HexgonalGrid3DGenerator:

    def __init__(self, polygon_border=None, polygon_obstacle=None, depth=None, neighbour_distance=0):
        self.depth = depth
        self.grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                                            distance_neighbour=neighbour_distance)
        self.coordinates2d = self.grid.coordinates2d
        self.get_3d_coordinates()

    def get_3d_coordinates(self):
        t1 = time.time()
        self.coordinates = []
        for i in range(len(self.depth)):
            for j in range(len(self.coordinates2d)):
                self.coordinates.append([self.coordinates2d[j, 0], self.coordinates2d[j, 1], self.depth[i]])
        self.coordinates = np.array(self.coordinates)
        t2 = time.time()
        print("3D coordianates are created successfully! Time consumed: ", t2 - t1)


