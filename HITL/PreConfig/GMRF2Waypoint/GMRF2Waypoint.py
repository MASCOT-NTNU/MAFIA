"""
This script generates hash table for gmrf & waypoint
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-23
"""


from Config.Config import *
from usr_func import *
import pickle


class HashGMRF2Waypoint:

    def __init__(self):
        self.load_waypoint()
        self.load_gmrf_grid()
        self.get_gmrf2waypoint_hash_tables()
        self.get_waypoint2gmrf_hash_tables()

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv")

    def get_gmrf2waypoint_hash_tables(self):
        self.gmrf2waypoint_hash_table = dict()

        t1 = time.time()
        for i in range(self.gmrf_grid.shape[0]):
            x = self.gmrf_grid['x'][i]
            y = self.gmrf_grid['y'][i]
            z = self.gmrf_grid['z'][i]
            ind = get_ind_at_location3d_xyz(self.waypoints.to_numpy(), x, y, z)
            self.gmrf2waypoint_hash_table[i] = ind
        t2 = time.time()

        filehandler = open(FILEPATH+'Config/HashGMRF2Waypoint.p', 'wb')
        with filehandler as f:
            pickle.dump(self.gmrf2waypoint_hash_table, f)
        f.close()
        print("Hashing finished, time consumed: ", t2 - t1)

    def get_waypoint2gmrf_hash_tables(self):
        self.waypoint2gmrf_hash_table = dict()

        t1 = time.time()
        for i in range(self.waypoints.shape[0]):
            x = self.waypoints['x'][i]
            y = self.waypoints['y'][i]
            z = self.waypoints['z'][i]
            ind = get_ind_at_location3d_xyz(self.gmrf_grid.to_numpy(), x, y, z)
            self.waypoint2gmrf_hash_table[i] = ind
        t2 = time.time()

        filehandler = open(FILEPATH+'Config/HashWaypoint2GMRF.p', 'wb')
        with filehandler as f:
            pickle.dump(self.waypoint2gmrf_hash_table, f)
        f.close()
        print("Hashing finished, time consumed: ", t2 - t1)

    def check_hash(self):
        self.error_gmrf2waypoint = []
        self.error_waypoint2gmrf = []
        for i in range(1000):
            ind = np.random.randint(self.waypoints.shape[0])
            loc_waypoint = self.waypoints.to_numpy()[ind]
            loc_gmrf = self.gmrf_grid.to_numpy()[self.waypoint2gmrf_hash_table[ind]]
            error_waypoint2gmrf = self.get_error(loc_waypoint, loc_gmrf)
            self.error_waypoint2gmrf.append(error_waypoint2gmrf)

            ind = np.random.randint(self.gmrf_grid.shape[0])
            loc_gmrf = self.gmrf_grid.to_numpy()[ind]
            loc_waypoint = self.waypoints.to_numpy()[self.gmrf2waypoint_hash_table[ind]]
            error_gmrf2waypoint = self.get_error(loc_waypoint, loc_gmrf)
            self.error_gmrf2waypoint.append(error_gmrf2waypoint)

    def get_error(self, loc1, loc2):
        error = np.sqrt((loc1[0] - loc2[0]) ** 2 +
                 (loc1[1] - loc2[1]) ** 2 +
                 (loc1[2] - loc2[2]) ** 2)
        return error

    def plot_mataching(self):
        # ind = np.random.randint(0, self.waypoints.shape[0], 1000)
        ind = np.arange(self.waypoints.shape[0])
        self.waypoints = self.waypoints.to_numpy()
        self.gmrf_grid = self.gmrf_grid.to_numpy()

        plt.figure(figsize=(30, 30))
        plt.plot(self.waypoints[ind, 1], self.waypoints[ind, 0], 'ro', label='waypoint')
        for i in ind:
            plt.plot(self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 1],
                     self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 0], 'gx')
        plt.plot(self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 1],
                 self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 0], 'gx', label='gmrf')
        plt.legend()
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.title("Waypoint 2 GMRF grid mapping illustration")
        plt.savefig(FIGPATH+"waypoint2gmrf.jpg")
        plt.show()

    def check_in_3d(self):
        ind = np.random.randint(0, self.waypoints.shape[0], 1000)
        self.waypoints = self.waypoints.to_numpy()
        self.gmrf_grid = self.gmrf_grid.to_numpy()
        fig = go.Figure(data=[go.Scatter3d(
            x=self.waypoints[ind, 1],
            y=self.waypoints[ind, 0],
            z=self.waypoints[ind, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='black',
                # colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        )])
        xgmrf = []
        ygmrf = []
        zgmrf = []
        for i in ind:
            xgmrf.append(self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 1])
            ygmrf.append(self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 0])
            zgmrf.append(self.gmrf_grid[self.waypoint2gmrf_hash_table[i], 2])
        fig.add_trace(go.Scatter3d(
            x=xgmrf,
            y=ygmrf,
            z=zgmrf,
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                # colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        ))
        plotly.offline.plot(fig, filename=FIGPATH+"waypoint2gmrf.html", auto_open=True)

if __name__ == "__main__":
    n = HashGMRF2Waypoint()
    n.check_hash()
    # n.check_in_3d()
    n.plot_mataching()





