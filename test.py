path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/models/"
from MAFIA.spde import spde
import matplotlib.pyplot as plt

import time

t = spde(model=2) #

t1 = time.time()
pos = [20, 0, 0]
var_cand = t.candidate(pos).reshape(45, 45, 11)
t2 = time.time()
print("time consumed: ", t2 - t1)

plt.imshow(var_cand[:, :, 0], origin="lower", vmin=0, vmax=1)
plt.colorbar()
plt.show()

#%%
samples = t.sample() + t.mu.reshape(-1, 1)# n ==

#%%


v = samples.reshape(45, 45, 10)
plt.imshow(v[:, :, 0], origin="lower", cmap="Paired", vmin=0, vmax=32)
plt.colorbar()

plt.show()

#%%
var = t.mvar().reshape(45, 45, 10)
plt.imshow(var[:, :, 9], origin="lower")
plt.colorbar()
plt.show()


#%%


#%%
pos = [22, 0, 0]
t.update(35, pos)

#%%

samples = t.sample() + t.mu.reshape(-1, 1)# n ==

#%%
import matplotlib.pyplot as plt
#%%

v = t.mu.reshape(45, 45, 10)
plt.imshow(v[:, :, 0], origin="lower", cmap="Paired", vmin=0, vmax=32)
plt.colorbar()
plt.show()

#%%
import numpy as np
lat = np.load('MAFIA/models/lats.npy')
lon = np.load('MAFIA/models/lons.npy')
depth = np.load('MAFIA/models/debth.npy')

#%%
# t1 = t.reshape(45, 45, 10)
plt.plot(lon, lat, 'k.')
plt.show()

#%%
from usr_func import *
NUM = 25
x = np.linspace(0, 1, NUM)
y = np.linspace(0, 1, NUM)
xx, yy = np.meshgrid(x, y)
xv, yv = map(vectorise, [xx, yy])
grid = np.hstack((xv, yv))
from scipy.spatial.distance import cdist
dist = cdist(grid, grid)
sigma = 1
eta = 4.5 / .5
Sigma = sigma ** 2 * (1 + eta * dist) * np.exp(-eta * dist)
plt.imshow(Sigma)
plt.colorbar()
plt.show()

def vectorise(x):
    return np.array(x).reshape(-1, 1)
