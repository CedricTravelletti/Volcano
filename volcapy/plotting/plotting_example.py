import volcapy.loading as ld
import numpy as np
import matplotlib.pyplot as plt
from volcapy.inverse.flow import InverseProblem


path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
data = ld.load_niklas(path)
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
coords = data['coords'].astype(dtype=np.float32, order='C', copy=False)

data = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m_post_wend_400m.npy")
data_100m = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m_post_wend_100m.npy")

import volcapy.plotting.plot as myplt

plt.plot_z_slice([0], data_100m.reshape(-1), coords)

# BASIS FUNCTIONS
data_basis_200m = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m_post_basis_200m.npy")
myplt.plot_z_slice([0], data_basis_200m.reshape(-1), coords)

data_basis_1000m = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m_post_basis_1000m.npy")
myplt.plot_z_slice([0], data_basis_1000m.reshape(-1), coords)


data_wnd_1000m = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m_post_wend_1000m.npy")
myplt.plot_z_slice([0], data_wnd_1000m.reshape(-1), coords)

data_wnd_100m = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m_post_wend_100m.npy")
myplt.plot_z_slice([0], data_wnd_100m.reshape(-1), coords)


# Training Evolution of Train/Test RMSE.
lls = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/log_likelihoods_contour.npy")
m0s = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/m0s_contour.npy")
sigma0s = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/sigma0s_contour.npy")
test_rmses = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/test_rmses_contour.npy")
train_rmses = np.load("/home/cedric/PHD/Dev/Volcano/volcapy/plotting/train_rmses_contour.npy")

f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(list(range(len(train_rmses))), train_rmses, label='train_rmse')
axarr[0].plot(list(range(len(train_rmses))), test_rmses, label='test_rmse')
axarr[0].axvline(x=2943.0)
axarr[0].legend()

axarr[1].plot(list(range(len(train_rmses))), lls)
axarr[1].axvline(x=2943.0)

axarr[2].plot(list(range(len(train_rmses))), m0s - 2000.0, label='m0 - 2000.0')
axarr[2].plot(list(range(len(train_rmses))), sigma0s, label='sigma0')
axarr[2].axvline(x=2943.0)
axarr[2].legend()
plt.xlabel("Training epoch")
plt.title("Evolution during training, fixed lambda0 = 200.0m, Test/Train split: 450/102")
plt.show()
