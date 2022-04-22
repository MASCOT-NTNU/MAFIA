"""
This script creates a hash table to save neighbouring location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from MAFIA.Simulation.Config.Config import *
from usr_func import *
import pickle


class HashNeighbourLocations:

    def __init__(self):
        self.load_coordinates()
        self.get_neighbours_hash_table()

    def load_coordinates(self):
        self.coordinates = pd.read_csv(FILEPATH + "Simulation/Field/Grid/Grid.csv").to_numpy()

    def get_neighbours_hash_table(self):
        self.neighbour_hash_table = dict()
        t1 = time.time()
        for i in range(self.coordinates.shape[0]):
            delta_x = self.coordinates[:, -3] - self.coordinates[i, -3]
            delta_y = self.coordinates[:, -2] - self.coordinates[i, -2]
            delta_z = self.coordinates[:, -1] - self.coordinates[i, -1]
            self.distance_euclidean = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
            self.distance_ellipsoid = (delta_x ** 2 / (1.5 * DISTANCE_LATERAL) ** 2) + \
                                      (delta_y ** 2 / (1.5 * DISTANCE_LATERAL) ** 2) + \
                                      (delta_z ** 2 / (DISTANCE_VERTICAL + 0.3) ** 2)
            self.ind_neighbour = np.where((self.distance_ellipsoid <= 1) * (self.distance_euclidean > DISTANCE_SELF))[0]
            self.neighbour_hash_table[i] = list(self.ind_neighbour)
        t2 = time.time()

        filehandler = open(FILEPATH+'/Simulation/Field/Grid/Neighbours.p', 'wb')
        with filehandler as f:
            pickle.dump(self.neighbour_hash_table, f)
        f.close()
        print("Hashing finished, time consumed: ", t2 - t1)

    def plot_neighbours(self, ind):
        # Helix equation
        t = np.linspace(0, 20, 100)
        x = self.coordinates[:, 1]
        y = self.coordinates[:, 0]
        z = self.coordinates[:, 2]
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=2,
                color='black',
                opacity=0.8
            )
        )])

        x = [self.coordinates[ind, 1]]
        y = [self.coordinates[ind, 0]]
        z = [self.coordinates[ind, 2]]
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8
            )
        ))

        x = self.coordinates[self.neighbour_hash_table[ind], 1]
        y = self.coordinates[self.neighbour_hash_table[ind], 0]
        z = self.coordinates[self.neighbour_hash_table[ind], 2]
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                # color=z,  # set color to an array/list of desired values
                # colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        ))

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        plotly.offline.plot(fig, filename=FIGPATH+"neighbour.html",
                            auto_open=True)

if __name__ == "__main__":
    n = HashNeighbourLocations()
    n.plot_neighbours(np.random.randint(n.coordinates.shape[0]))







