import matplotlib.pyplot as plt
import numpy as np


lls = np.load("lls_contour.npy")
train_rmses = np.load("train_rmses_contour.npy")
m0s = np.load("m0s_contour.npy")
sigma0s = np.load("sigma0s_contour.npy")


plt.contour(sigma0s, m0s, lls)
plt.contour(sigma0s, m0s, train_rmses)
plt.show()

plot = plt.contour(sigma0s, m0s, lls, levels=[-1200, -1180, -1171, -1170,
        -1160, -1150, -1000, 0, 500, 1000, 4000])
plt.clabel(plot, inline=1, fontsize=7)
# plt.contourf(sigma0s, m0s, train_rmses)
plt.show()

plot = plt.contour(sigma0s, m0s, train_rmses, levels=[0.1, 0.2,
        0.5])
plt.clabel(plot, inline=1, fontsize=7)
plt.show()
