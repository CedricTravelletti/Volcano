""" Load helper script. Makes basic Niklas data available.
Used for fast prototyping.

"""
import numpy as np

from volcapy.loading import load_niklas
# ------------------------------------------------------------------------

# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)


F = niklas_data['F']
n_data = F.shape[0]
n_model = F.shape[1]

d_obs = niklas_data['d']

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
coords_x = np.ascontiguousarray(coords[:,0])
coords_y = np.ascontiguousarray(coords[:,1])
coords_z = np.ascontiguousarray(coords[:,2])
