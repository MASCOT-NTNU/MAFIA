# MAFIA
---
MAFIA is used to construct the mean and precision matrix of a 3D Gaussian Markov Random Field (GMRF) in a specific volume in the Trondheimsfjord for intelligent ocean sampling using Autonomous Underwater Vehicles (AUVs).
The GMRFs are approximated through a Stochastic Partial differential equation with the matérn covariance function as its stationary solution.
We have parameterized three different models; A stationary model with vertical-lateral anisotropy, a similar Non-stationary, and a Non-stationary model with bidirectional anisotropy, and have fit them to a field approximation from a complex numerical ocean model (SINMOD).
The precision matrix is found in `./models/SINMOD-SA.npy`, `./models/SINMOD-NI.npy`, and `./models/SINMOD-NA.npy` respectively, and the standard devitation of Gaussian white noise in the field and the measurement noise of the AUV is found in `./models/sigma.npy`.

It is implemented as a class so it can simply loaded with:
```python
from spde import spde
```

To load the different models simply specify it in the class constructor:

### Stationary model with vertical-laterial anisotropy

```python
mod = spde(model == 1)
```

### Non-Stationary model with vertical-laterial anisotropy

```python
mod = spde(model == 2)
```

### Non-Stationary model with vertical-laterial anisotropy

```python
mod = spde(model == 3)
```

## General functionalities
To sample from the model do the call:
```python
mod.sample(n=1)
```
where `n` specifies the number of realizations of the entire field.

To estimate the marginal variance of the precision matrix:
```python
mod.mvar()
```
This is estimated empricically with the samples from the model.

To get the candidate variance given that you have sampled in a point `pos` within the field:
```python
mod.candidate(pos = [xi,yi,zi])
```
This can be used to find the points which are most informative.

When the AUV has sampled a point it can update the field with:

```python
mod.search_path_from_trees(rel=measurement, pos=[xi, yi, zi])
```
This updates the mean and precision matrix of the field within the class and returns nothing.

To get attributees from the class

```python
mod.mu  # the mean of the field (20250) vector
mod.Q  # the precision matrix of the field (20250x20250) sparse matrix
mod.sigma  # gaussian noise of the field (mod.sigma[1]) and the measurement noise of the AUV (mod.sigma[0])
mod.Q_fac  # the choleksy factorized Q. This is a object returned from the C library cholmod
mod.lats  # latitude boundary of the field
mod.lons  # longitude boundary of the field
mod.X_START  # x boundary of the field
mod.Y_START  # y boundary of the field
```

# HITL of MAFIA
---
Open 4 iterfaces either through tmux or `Ctrl+Alt+T`.
---
- `cd ~/catkin_ws/`
- `source devel/setup.bash`
---
- `cd ~/dune_all/build`
- `./dune -c lauv-simulator-1 -p Simulation`
---
- `cd ~/catkin_ws/`
- `source devel/setup.bash`
- `roslaunch src/imc_ros_interface/launch/bridge.launch `
---
- `cd HITL/`
- `python3 MAFIALauncher.py`
---
Then go to neptus to activate follow reference.
