
from usr_func import *

from MAFIA.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from MAFIA.Simulation.Kernel.Kernel import Kernel
from MAFIA.Simulation.Field.Grid.Location import *
from MAFIA.Simulation.Field.Knowledge.Knowledge import Knowledge
from MAFIA.spde import spde
import pickle

FILEPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/"


coordinates = pd.read_csv(FILEPATH + "Simulation/Field/Grid/Grid.csv").to_numpy()
neighbour_hash_table_filehandler = open(FILEPATH + "Simulation/Field/Grid/Neighbours.p", 'rb')
neighbour_hash_table = pickle.load(neighbour_hash_table_filehandler)
neighbour_hash_table_filehandler.close()


current_location = Location(63.45121, 10.40673, .5)

knowledge = Knowledge(coordinates=coordinates, neighbour_hash_table=neighbour_hash_table, threshold=THRESHOLD,
                      spde_model=spde, previous_location=current_location, current_location=current_location)

t = MyopicPlanning3D(knowledge=knowledge)
# t.knowledge.ind_current_location





