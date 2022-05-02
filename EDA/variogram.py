import matplotlib.pyplot as plt

from usr_func import *
lat_origin = 63.3
lon_origin = 10.4

dim = 3
angles = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
# model = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=angles)
# x = y = z = range(50)
FILEPATH = "/Users/yaolin/HomeOffice/MAFIA/EDA/"
FIGPATH = FILEPATH + "fig/"
data_auv = pd.read_csv(FILEPATH + "auv.csv").to_numpy()
data_sinmod = pd.read_csv(FILEPATH + "sinmod.csv").to_numpy()

lat_auv = data_auv[:, 0]
lon_auv = data_auv[:, 1]
depth_auv = data_auv[:, 2]
sal_auv = data_auv[:, 3]
sal_sinmod = data_sinmod[:, 3]
x_auv, y_auv = latlon2xy(lat_auv, lon_auv, lat_origin, lon_origin)
z_auv = depth_auv
sal_residual = sal_auv - sal_sinmod

def get_xyzs_at_depth(x, y, z, s, depth=.5):
    ind = np.where((z<depth+.25)*(z>depth-.25))[0]
    return x[ind], y[ind], z[ind], s[ind], ind

depth = .5
xr, yr, zr, sr, ind_auv = get_xyzs_at_depth(x_auv, y_auv, z_auv, sal_residual, depth=depth)
V = Variogram(coordinates=np.vstack((yr, xr)).T, values=sr, use_nugget=True, model='Matern',
              n_lags=20, maxlag=1000)
lateral_range = V.cof[0]
sill = V.cof[1]
nugget = V.cof[2]
print("range: ", lateral_range)
print("sill: ", sill)
print("nugget: ", nugget)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(vectorise(sal_sinmod[ind_auv]), vectorise(sal_auv[ind_auv]))
beta0 = model.intercept_
beta1 = model.coef_

fig = plt.figure(figsize=(50, 10))
gs = GridSpec(nrows=1, ncols=4)
ax = fig.add_subplot(gs[0])
im = ax.scatter(yr, xr, c=sr, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(im)
ax.text(0.1, 0.9, 'beta0: {:.2f}'.format(beta0[0]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.1, 0.8, 'beta1: {:.2f}'.format(beta1[0, 0]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.1, 0.7, 'residual: sal_auv - sal_sinmod', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Residual")

ax = fig.add_subplot(gs[1])
ax.plot(V.bins, V.experimental, 'k.')
z = np.polyfit(V.bins, V.experimental, 2)
f = np.poly1d(z)
ax.plot(np.linspace(np.min(V.bins), np.max(V.bins), 100), f(np.linspace(np.min(V.bins), np.max(V.bins), 100)), 'r-')
ax.text(0.1, 0.9, 'Range: {:.2f} [m]'.format(lateral_range), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.1, 0.8, 'Sill (sigma**2 + nugget): {:.2f}'.format(sill), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.1, 0.7, 'Nugget: {:.2f}'.format(nugget), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.set_title("Empirical variogram")

ax = fig.add_subplot(gs[2])
im = ax.scatter(yr, xr, c=sal_auv[ind_auv], cmap='BrBG', vmin=10, vmax=30)
plt.colorbar(im)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Salinity[psu]")

ax = fig.add_subplot(gs[3])
im = ax.scatter(yr, xr, c=sal_sinmod[ind_auv], cmap="BrBG", vmin=10, vmax=30)
# im = ax.scatter(y_sinmod, x_sinmod, c=ave_salinity_data_sinmod_raw[ind_sinmod_depth, :, :], cmap="BrBG", vmin=10, vmax=30)
plt.colorbar(im)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Salinity[psu]")
plt.savefig(FIGPATH+"d_{:02d}.jpg".format(int(depth)))
plt.show()

#%%

data_sinmod_raw = netCDF4.Dataset("/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2021.05.27.nc")
ref_timestamp = datetime.strptime("2021.05.27", "%Y.%m.%d").timestamp()
timestamp = np.array(data_sinmod_raw["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp
lat_data_sinmod_raw = np.array(data_sinmod_raw['gridLats'])
lon_data_sinmod_raw = np.array(data_sinmod_raw['gridLons'])
depth_data_sinmod_raw = np.array(data_sinmod_raw['zc'])
salinity_data_sinmod_raw = np.array(data_sinmod_raw['salinity'])
ave_salinity_data_sinmod_raw = np.mean(salinity_data_sinmod_raw, axis=0)
x_sinmod, y_sinmod = latlon2xy(lat_data_sinmod_raw, lon_data_sinmod_raw, lat_origin, lon_origin)


fig = plt.figure(figsize=(50, 10))
gs = GridSpec(nrows=1, ncols=4)
ax = fig.add_subplot(gs[0])
im = ax.scatter(y_sinmod, x_sinmod, c=salinity_data_sinmod_raw[0, 0, :, :], cmap='BrBG', vmin=10, vmax=30)
plt.colorbar(im)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Time frame: 0")

ax = fig.add_subplot(gs[1])
im = ax.scatter(y_sinmod, x_sinmod, c=salinity_data_sinmod_raw[50, 0, :, :], cmap='BrBG', vmin=10, vmax=30)
plt.colorbar(im)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Time frame: 50")

ax = fig.add_subplot(gs[2])
im = ax.scatter(y_sinmod, x_sinmod, c=salinity_data_sinmod_raw[100, 0, :, :], cmap='BrBG', vmin=10, vmax=30)
plt.colorbar(im)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Time frame: 100")

ax = fig.add_subplot(gs[3])
im = ax.scatter(y_sinmod, x_sinmod, c=salinity_data_sinmod_raw[-1, 0, :, :], cmap='BrBG', vmin=10, vmax=30)
plt.colorbar(im)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title("Time frame: 0")
plt.savefig(FIGPATH+"d_{:02d}.jpg".format(int(depth)))
plt.show()

