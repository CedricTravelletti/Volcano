""" Pre-analysis script
"""
import numpy as np
import h5py
import plotly as plt
import plotly.graph_objs as go


data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
dataset = h5py.File(data_path)

# Load the DTM.
x = np.array(dataset['x'])[:, 0]
y = np.array(dataset['y'])[:, 0]
z = np.array(dataset['z'])

# Delete below sea level.
z[z < 0.0] = 0.0

# Unravel z.
x_unravelled = []
y_unravelled = []
z_unravelled = []

for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        x_unravelled.append(x[i])
        y_unravelled.append(y[j])
        z_unravelled.append(z[i, j])

# Load the locations of the measures.
lat = np.array(f['lat'])[0, :]
lon = np.array(f['long'])[0, :]
height = np.array(f['GPS_ground'])[0, :]

# Measures plot.
measures_plot = go.Scatter3d(
        x=lat,
        y=lon,
        z=height,
        mode='markers',
        marker=dict(
                size=2)
        )

topography_plot = go.Scattergl(
        x=x_unravelled[0:800000],
        y=y_unravelled[0:800000],
        z=z_unravelled[0:800000],
        mode='markers',
        marker=dict(
                size=2)
        )

plt.offline.plot([topography_plot])

layout = go.Layout(
        title = "flj",
        autosize = False,
        width = 2000,
        height = 1000,
        scene = dict(
                aspectmode = 'cube',
                xaxis = dict(
                        range = [300, 700]),
                yaxis = dict(
                        range = [300, 700]),
                zaxis = dict(
                        range = [0, 1000])))

fig = go.Figure(data=[go.Surface(z = z)],
        layout=layout)

plt.offline.plot(fig)
