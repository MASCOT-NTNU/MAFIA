
"""
This script tests comparison
"""
from usr_func import *
from MAFIA.Simulation.Config.Config import *

waypoint_graph = pd.read_csv(FILEPATH+"Simulation/Config/WaypointGraph.csv").to_numpy()
gmrf_grid = pd.read_csv(FILEPATH+"Simulation/Config/GMRFGrid.csv").to_numpy()

ind_top_gmrf = np.where(gmrf_grid[:, 2] < 0)[0]
ind_top_waypoint = np.where(waypoint_graph[:, 2] < 1)[0]

plt.plot(gmrf_grid[ind_top_gmrf, 1], gmrf_grid[ind_top_gmrf, 0], 'k.', alpha=.4)
plt.plot(waypoint_graph[ind_top_waypoint, 1], waypoint_graph[ind_top_waypoint, 0], 'r.', alpha=.4)
plt.show()




