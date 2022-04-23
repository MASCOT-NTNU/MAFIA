from MAFIA.Simulation.Config.Config import *
from usr_func import *

gmrf_grid = pd.read_csv(FILEPATH+"Simulation/Config/GMRFGrid.csv").to_numpy()
sal = pd.read_csv(FILEPATH+"Simulation/Config/Data/data_mu_truth.csv")

fig = go.Figure(data=[go.Scatter3d(
    x=gmrf_grid[:, 1],
    y=gmrf_grid[:, 0],
    z=-gmrf_grid[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color=sal['salinity'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        cmin=0,
        cmax=30,
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename=FIGPATH+"data.html", auto_open=True)


# fig = go.Figure(data=[go.Scatter3d(
#     x=sal['lon'],
#     y=sal['lat'],
#     z=-sal['depth'],
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=sal['salinity'],                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         cmin=0,
#         cmax=30,
#         opacity=0.8
#     )
# )])
#
# # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# plotly.offline.plot(fig, filename=FIGPATH+"data.html", auto_open=True)


