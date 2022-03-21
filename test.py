path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/MAFIA/models/"
from spde import spde
import matplotlib.pyplot as plt

import time

t = spde(model=2) #

t1 = time.time()
pos = [40, 0, 0]
var_cand = t.candidate(pos).reshape(45, 45, 10)
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
t = np.load('MAFIA/models/-NI.npy')

#%%
# t1 = t.reshape(45, 45, 10)






