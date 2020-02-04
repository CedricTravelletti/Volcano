""" (temporary) Convert the results of sequential of synthetic data to vtk
(should have been done directly in the script).

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.synthetic.vtkutils import save_vtk


def main():
    # Load
    data_folder = "../synthetic/synthetic_data/"
    F = np.load(os.path.join(data_folder, "F_synth.npy"))
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

    for i in range(20):
        m = np.load("m_post_{}.npy".format(i))

        # Save to VTK format..
        # Have to put back in rectangular grid.
        m_post_reg = np.zeros(reg_coords.shape[0])
        m_post_reg[volcano_inds] = m
        save_vtk(m_post_reg, (nx, ny, nz), res_x, res_y, res_z,
                "m_post_{}.mhd".format(i))

if __name__ == "__main__":
    main()
