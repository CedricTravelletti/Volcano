import matplotlib.pyplot as plt
import numpy as np


lls = np.load("lls_contour.npy")
train_rmses = np.load("train_rmses_contour.npy")
m0s = np.load("m0s_contour.npy")
sigma0s = np.load("sigma0s_contour.npy")


plt.contour(sigma0s, m0s, lls)
plt.contour(sigma0s, m0s, train_rmses)
plt.show()
