""" Plot evolution of test / train RMSE for different lambda0s.
This can be used to visualize results when doing brute for search on lambda0.

"""
import volcapy.loading as ld
import numpy as np
import os
import matplotlib.pyplot as plt
from volcapy.inverse.flow import InverseProblem

"""
LOAD
"""
# input_folder = "/idiap/temp/ctravelletti/out/simple_exponential/"
input_folder = "/home/cedric/PHD/run_results/matern/"

# Training Evolution of Train/Test RMSE for multiple lambdas.
lambda0s = np.load(os.path.join(input_folder, "lambda0s_train.npy"))
lls = np.load(os.path.join(input_folder, "log_likelihoods_train.npy"))
m0s = np.load(os.path.join(input_folder, "m0s_train.npy"))
sigma0s = np.load(os.path.join(input_folder, "sigma0s_train.npy"))
test_rmses = np.load(os.path.join(input_folder, "test_rmses_train.npy"))
train_rmses = np.load(os.path.join(input_folder, "train_rmses_train.npy"))

# Plot Evolution
plt.figure()
plt.subplot(211)
plt.title("Train/Test split: 450/103.")
plt.plot(lambda0s[:], train_rmses[:, 149], "r*", label="Train")
plt.plot(lambda0s[:], test_rmses[:, 149], "bo", label="Test")
plt.xlabel("lambda0 [m]")
plt.ylabel("RMSE [mGal], std=0.1")
plt.axhline(0.1, color='r')
plt.legend()

plt.subplot(212)
plt.plot(lambda0s[:], lls[:, 149])
plt.xlabel("lambda0 [m]")
plt.ylabel("Log-likelihood (shifted)")
plt.show()


# Plot Parameters
plt.figure()
plt.subplot(211)
plt.title("Train/Test split: 450/103.")
plt.plot(lambda0s[:], m0s[:, 149], "r*", label="m0")
plt.xlabel("lambda0 [m]")
plt.ylabel("m0 [kg/m3]")
plt.axhline(0.1, color='r')
plt.legend()

plt.subplot(212)
plt.plot(lambda0s[:], sigma0s[:, 149], "bo")
plt.xlabel("lambda0 [m]")
plt.ylabel("sigma0 [kg/m3]")
plt.show()

"""
Description of data format:
---------------------------
We have different lambdas.

Each array is indexed by lambda along the first axis, and by training step
along the second axis.
"""

# Discard the ones with nan log-likelihood.
non_nan_inds = list(range(9)) + [12]

# Plot params RMSES.
f, axarr = plt.subplots(5, sharex=True)
axarr[0].set_title("Evolution of parameters and test RMSE along training for multiple lambda0s. Train/Test: 450/102.")

# Loop over lambdas.
for i, lmbda in enumerate(lambda0s):
    axarr[0].plot(m0s[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[0].set_xlabel("m0")

    axarr[1].plot(sigma0s[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[1].set_xlabel("sigma0")

    axarr[2].plot(train_rmses[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[2].set_xlabel("train_rmse")
    axarr[2].set_ylim((0.0, 0.2))

    axarr[3].plot(test_rmses[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[3].set_xlabel("test_rmse")
    axarr[3].set_ylim((0.0, 0.2))

    axarr[4].plot(lls[i, :], label="lambda0 = {}".format(lambda0s[i]))
    axarr[4].set_xlabel("log-likelihood")
plt.legend()
plt.show()

"""
# Plot final log likelihood.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title("Evolution of parameters and test RMSE along training for multiple lambda0s. Train/Test: 450/102.")
axarr[0].plot(lambda0s[:9], m0s[:9, -1])
axarr[0].set_xlabel("log-likelihood")

axarr[1].plot(lambda0s[:9], test_rmses[:9, -1])
axarr[1].set_xlabel("test RMSE")

plt.legend()
plt.show()
"""
