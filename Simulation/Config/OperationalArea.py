"""
This class will get the operational area
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""

import geopandas
import pandas as pd
import shapely
import shapely.geometry.polygon
from shapely.geometry import Polygon, GeometryCollection
from usr_func import *
from rdp import rdp
BUFFER_SIZE_BORDER = -100 # [m]
BUFFER_SIZE_MUNKHOLMEN = 50 # [m]

'''
Path
'''
PATH_SHAPE_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/GIS/Munkholmen.shp" # remember to copy all the
# other files in as well
BOUNDARY_SHAPE_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/GIS/boundary.csv"

FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/Simulation/Config/"
PATH_OPERATION_AREA = FILEPATH + "OperationalArea.csv"
# PATH_MUNKHOLMEN = FILEPATH + "Munkholmen.csv"

''' Note
Sometimes the geometry will have multiple polygons due to merging of different parts as the result of the buffer size
Be careful with the check conditions
'''


class OpArea:

    def __init__(self):
        self.get_operational_area()
        # self.save_operational_areas()
        pass

    def get_operational_area(self):
        self.munkholmen_shape_file = geopandas.read_file(PATH_SHAPE_FILE)
        self.boundary_shape_file = np.fliplr(pd.read_csv(BOUNDARY_SHAPE_FILE)) # swap x & y
        self.polygon_path_boundary = Polygon(self.boundary_shape_file)

        # get big border polygon
        self.Trondheimsfjord = self.munkholmen_shape_file[self.munkholmen_shape_file['name'] == "Trondheimsfjorden"]['geometry']
        self.polygon_path_trondheimsfjorden = self.Trondheimsfjord.to_numpy()[0][0]
        self.lon_fjord = vectorise(self.polygon_path_trondheimsfjorden.exterior.xy[0])
        self.lat_fjord = vectorise(self.polygon_path_trondheimsfjorden.exterior.xy[1])
        self.polygon_fjord = np.hstack((self.lat_fjord, self.lon_fjord))
        self.polygon_fjord_buffered = self.get_buffered_polygon(self.polygon_fjord, BUFFER_SIZE_BORDER,
                                                                which_polygon=1, plot=False)
        # plt.plot(self.polygon_fjord[:, 1], self.polygon_fjord[:, 0], 'k-')
        # plt.plot(self.polygon_fjord_buffered[:, 1], self.polygon_fjord_buffered[:, 0], 'b-')
        # plt.show()

        # get munkholmen obstacle polygon
        # self.Munkholmen = self.munkholmen_shape_file[self.munkholmen_shape_file['name'] == "Munkholmen"]['geometry']
        # self.polygon_path_munkholmen = self.Munkholmen.to_numpy()[0]
        # self.lon_munkholmen = vectorise(self.polygon_path_munkholmen.exterior.xy[0])
        # self.lat_munkholmen = vectorise(self.polygon_path_munkholmen.exterior.xy[1])
        # self.polygon_munkholmen = np.hstack((self.lat_munkholmen, self.lon_munkholmen))


        self.intersection = []
        self.intersection.append(self.get_intersected_polygons(Polygon(np.fliplr(self.polygon_fjord_buffered)),
                                                               self.polygon_path_boundary))
        # # self.intersection.append(self.get_intersected_polygons(self.polygon_path_trondheimsfjorden,
        # #                                                        self.polygon_path_munkholmen))

        self.operational_regions = GeometryCollection(self.intersection)
        self.operational_areas = self.operational_regions.geoms
        self.lon_operational_area = vectorise(self.operational_areas[0].exterior.xy[0])
        self.lat_operational_area = vectorise(self.operational_areas[0].exterior.xy[1])
        self.polygon_operational_area = np.hstack((self.lat_operational_area, self.lon_operational_area))

        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.lon_operational_area, self.lat_operational_area, 'k-.', linewidth=2)
        plt.plot(self.boundary_shape_file[:, 0], self.boundary_shape_file[:, 1], 'r-')
        plt.show()

        df = pd.DataFrame(self.polygon_operational_area, columns=['lat', 'lon'])
        df.to_csv(FILEPATH+"Polygon_border.csv", index=False)

    def get_intersected_polygons(self, polygon1, polygon2):
        return polygon1.intersection(polygon2)

    def get_buffered_polygon(self, polygon, buffer_size, which_polygon=1, plot=True):
        x, y = latlon2xy(polygon[:, 0], polygon[:, 1], 0, 0)
        polygon_xy = np.hstack((vectorise(x), vectorise(y)))
        polygon_xy_shapely = Polygon(polygon_xy)
        polygon_xy_shapely_buffered = polygon_xy_shapely.buffer(buffer_size)
        if type(polygon_xy_shapely_buffered) == shapely.geometry.polygon.Polygon:
            x_buffer, y_buffer = polygon_xy_shapely_buffered.exterior.xy

            # == shorten polygons using Ramer-Douglas-Peucker Algorithm
            # polygon_xy_buffer = np.hstack((vectorise(x_buffer), vectorise(y_buffer)))
            # polygon_xy_buffer_shorten = rdp(polygon_xy_buffer, epsilon=epsilon)
            # lat_wgs_shorten, lon_wgs_shorten = xy2latlon(polygon_xy_buffer_shorten[:, 0],
            #                                              polygon_xy_buffer_shorten[:, 1],
            #                                              0, 0)
            lat_wgs, lon_wgs = xy2latlon(vectorise(x_buffer), vectorise(y_buffer), 0, 0)
            polygon_wgs_buffer_shorten = np.hstack((vectorise(lat_wgs), vectorise(lon_wgs)))
            if plot:
                plt.plot(lon_wgs, lat_wgs, 'k.-')
                plt.plot(polygon[:, 1], polygon[:, 0], 'r.-')
                plt.show()
            return polygon_wgs_buffer_shorten
        else:
            for i in range(len(polygon_xy_shapely_buffered)):
                x_buffer, y_buffer = polygon_xy_shapely_buffered[i].exterior.xy

                # == shorten polygons using Ramer-Douglas-Peucker Algorithm
                # polygon_xy_buffer = np.hstack((vectorise(x_buffer), vectorise(y_buffer)))
                # polygon_xy_buffer_shorten = rdp(polygon_xy_buffer, epsilon=epsilon)
                # lat_wgs_shorten, lon_wgs_shorten = xy2latlon(polygon_xy_buffer_shorten[:, 0],
                #                                              polygon_xy_buffer_shorten[:, 1],
                #                                              0, 0)
                lat_wgs, lon_wgs = xy2latlon(vectorise(x_buffer), vectorise(y_buffer), 0, 0)
                polygon_wgs_buffer_shorten = np.hstack((vectorise(lat_wgs), vectorise(lon_wgs)))
                if plot:
                    plt.plot(lon_wgs, lat_wgs, 'k.-')
                    plt.plot(polygon[:, 1], polygon[:, 0], 'r.-')
                    plt.show()
                if i == which_polygon:
                    return polygon_wgs_buffer_shorten


if __name__ == "__main__":
    op = OpArea()

