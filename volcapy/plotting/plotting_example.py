import volcapy.loading as ld
import numpy as np


path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
data = ld.load_niklas(path)
coords = data['coords'].astype(dtype=np.float32, order='C', copy=False)

data = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/test_out/m_posterior.npy")

import volcapy.plotting.plot as plt

plt.plot_z_slice([-150, 0, 500, 800], data, coords[:, 0], coords[:, 1], coords[:, 2])
