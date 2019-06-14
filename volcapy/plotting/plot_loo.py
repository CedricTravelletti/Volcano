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
input_folder = "/home/cedric/PHD/run_results/cpu/"

# Training Evolution of Train/Test RMSE for multiple lambdas.
lambda0s = np.load(os.path.join(input_folder, "lambda0s_train.npy"))
lls = np.load(os.path.join(input_folder, "log_likelihoods_train.npy"))
m0s = np.load(os.path.join(input_folder, "m0s_train.npy"))
sigma0s = np.load(os.path.join(input_folder, "sigma0s_train.npy"))
train_rmses = np.load(os.path.join(input_folder, "train_rmses_train.npy"))
loocv_rmses = np.load(os.path.join(input_folder, "loocv_rmses_train.npy"))

# Plot Evolution
plt.figure()
plt.subplot(211)
plt.title("Train/Test split: 450/103.")
plt.plot(lambda0s, train_rmses, "r*", label="Train")
plt.plot(lambda0s, loocv_rmses, "bo", label="LOOCV RMSE")

# Have to add limits because of diverged trainings.
plt.ylim((0, 1.0))
plt.xlabel("lambda0 [m]")
plt.ylabel("RMSE [mGal], std=0.1")
plt.axhline(0.1, color='r')
plt.legend()

plt.subplot(212)
plt.plot(lambda0s, lls)
plt.xlabel("lambda0 [m]")
plt.ylabel("Log-likelihood (shifted)")
plt.show()


# Plot Parameters
plt.figure()
plt.subplot(211)
plt.title("Train/Test split: 450/103.")
plt.plot(lambda0s, m0s, "r*", label="m0")
plt.xlabel("lambda0 [m]")
plt.ylabel("m0 [kg/m3]")
plt.axhline(0.1, color='r')
plt.legend()

plt.subplot(212)
plt.plot(lambda0s, sigma0s, "bo")
plt.xlabel("lambda0 [m]")
plt.ylabel("sigma0 [kg/m3]")
plt.show()
