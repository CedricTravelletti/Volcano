import volcapy.loading as ld
import numpy as np
import matplotlib.pyplot as plt
from volcapy.inverse.flow import InverseProblem


# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
data = ld.load_niklas(niklas_data_path)
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

# Training Evolution of Train/Test RMSE for multiple lambdas.
lambda0s = np.load("/idiap/temp/ctravelletti/tflow/Volcano/volcapy/torch/presentation/lambda0s_train.npy")
lls = np.load("/idiap/temp/ctravelletti/tflow/Volcano/volcapy/torch/presentation/log_likelihoods_train.npy")
m0s = np.load("/idiap/temp/ctravelletti/tflow/Volcano/volcapy/torch/presentation/m0s_train.npy")
sigma0s = np.load("/idiap/temp/ctravelletti/tflow/Volcano/volcapy/torch/presentation/sigma0s_train.npy")
test_rmses = np.load("/idiap/temp/ctravelletti/tflow/Volcano/volcapy/torch/presentation/test_rmses_train.npy")
train_rmses = np.load("/idiap/temp/ctravelletti/tflow/Volcano/volcapy/torch/presentation/train_rmses_train.npy")

# Discard the ones with nan log-likelihood.
non_nan_inds = list(range(9)) + [12]
# Plot test RMSES.
for i in non_nan_inds:
    plt.plot(test_rmses[i, :], label="lambda0 = {}".format(lambda0s[i]))
plt.legend()
plt.ylim((0.0, 2.0))
plt.show()

# Plot params RMSES.
f, axarr = plt.subplots(4, sharex=True)
axarr[0].set_title("Evolution of parameters and test RMSE along training for multiple lambda0s. Train/Test: 450/102.")
for i in non_nan_inds:
    axarr[0].plot(m0s[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[0].set_xlabel("m0")

    axarr[1].plot(sigma0s[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[1].set_xlabel("sigma0")

    axarr[2].plot(test_rmses[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[2].set_xlabel("test_rmse")
    axarr[2].set_ylim((0.0, 2.0))

    axarr[3].plot(lls[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[2].set_xlabel("log-likelihood")
plt.legend()
plt.show()
