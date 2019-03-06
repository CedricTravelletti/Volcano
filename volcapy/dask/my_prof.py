""" Block matrix operations in TensorFlow.
Cpoied from

http://pages.cs.wisc.edu/~akella/CS838/F16/assignment3/uploaded-scripts/exampleMatmulSingle.py

"""
"""
A solution to finding trace of square of a large matrix using a single device.
We are able to circumvent OOM errors, by generating sub-matrices. TensorFlow
runtime, is able to schedule computation on small sub-matrices without
overflowing the available RAM.
"""
import numpy as np
import dask.array as da
import os
from timeit import default_timer as timer

from volcapy.loading import load_niklas
# ------------------------------------------------------------------------

# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)


F = niklas_data['F']
n_data = F.shape[0]
n_model = F.shape[1]
model_chunk_size = 10000
F = da.from_array(F, chunks=(n_data, n_model))

d_obs = niklas_data['d']
d_obs = da.from_array(d_obs, chunks=1000)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
coords_x = np.ascontiguousarray(coords[:,0])
coords_y = np.ascontiguousarray(coords[:,1])
coords_z = np.ascontiguousarray(coords[:,2])


def my_func(coords_part):
    x1, x2 = np.meshgrid(coords_part, coords_x, indexing='ij')
    diff_x = np.subtract(x1, x2) return diff_x
