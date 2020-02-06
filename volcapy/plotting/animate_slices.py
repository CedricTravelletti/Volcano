""" Produce slices showing the progress of sequential inversion.

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import volcapy.covariance.exponential as cl
from volcapy.synthetic.vtkutils import save_vtk


def main():
    # Load
    # data_folder = "../synthetic/synthetic_data/"
    data_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/synthetic_data/"
    # F = np.load(os.path.join(data_folder, "F_synth.npy"))
    reg_coords = np.load(os.path.join(data_folder,"reg_coords_synth.npy"))
    volcano_inds = np.load(os.path.join(data_folder,"volcano_inds_synth.npy"))
    ground_truth = np.load(os.path.join(data_folder,"density_synth.npy"))

    # Synthetic volcano grid parameters.
    nx = 120
    ny = 120
    nz = 60
    res_x = 50
    res_y = 50
    res_z = 50

    # Convert ground truth to 3d.
    gt_reg_3d = ground_truth.reshape((nx, ny, nz), order="C") # Back to 3d.
    # Slice near bottom.
    gt_slice = gt_reg_3d[:, :, 1]

    # for i in range(20):
    def animate(i):
        fig.suptitle("Recovery at batch {}/20.".format(i+1), fontsize=16)
        m = np.load("./data/m_post_{}.npy".format(i))

        # Project back to regular grid.
        m_reg = np.zeros(reg_coords.shape[0])
        m_reg[volcano_inds] = m

        # Back to 3d.
        m_reg_3d = m_reg.reshape((nx, ny, nz), order="C") # Back to 3d.
        # Slice near bottom.
        m_slice = m_reg_3d[:, :, 1]

        misfit = m_slice - gt_slice

        plt.subplot(131)
        plt.imshow(m_slice, cmap="seismic", vmin=900, vmax=2200)
        plt.title("Recovery")
        plt.subplot(132)
        plt.imshow(gt_slice, cmap="seismic", vmin=900, vmax=2200)
        plt.title("Ground Truth")
        plt.subplot(133)
        plt.imshow(misfit, cmap="jet", vmin=-300, vmax=300)
        plt.title("Misfit")

    fig = plt.figure()
    # Frame count starts at 1.
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=range(20), repeat=True)
    plt.show()

if __name__ == "__main__":
    main()
