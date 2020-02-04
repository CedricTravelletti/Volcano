""" Try the sequential inversion framework on synthetic data (conic volcano).

The synthetic data should be located in ../synthetic/synthetic_data/.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.synthetic.vtkutils import save_vtk


def main():
    # Load
    data_folder = "../synthetic/synthetic_data/"
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_synth.npy"))).float()
    reg_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"reg_coords_synth.npy"))).float()
    volcano_inds = torch.from_numpy(
            np.load(os.path.join(data_folder,"volcano_inds_synth.npy")))
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"data_coords_synth.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"data_values_synth.npy"))).float()
    ground_truth = torch.from_numpy(
            np.load(os.path.join(data_folder,"density_synth.npy"))).float()

    # List of cells belonging to the volcano (same as cells_coords in the
    # Stromboli setup). Here we extract them from a regular grid.
    volcano_coords = reg_coords[volcano_inds]
    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*8 / 1e9
    print("Size of Pushforward matrix: {} GB.".format(size))

    # Synthetic volcano grid parameters.
    nx = 120
    ny = 120
    nz = 60
    res_x = 50
    res_y = 50
    res_z = 50

    # Partition the data.
    # Test-Train split.
    n_keep = 300
    F_part_1 = F[:n_keep, :]
    F_part_2 = F[n_keep:600, :]

    data_1 = data_values[:n_keep]
    data_2 = data_values[n_keep:600]
    
    # Params
    data_std = 0.1
    lambda0 = 200.0
    sigma0 = 200.0
    m0 = 1300

    # Now ready to go to updatable covariance.
    from volcapy.update.updatable_covariance import UpdatableCovariance
    updatable_cov = UpdatableCovariance(cl, lambda0, sigma0, volcano_coords)

    from volcapy.update.updatable_covariance import UpdatableMean
    updatable_mean = UpdatableMean(m0 * torch.ones(volcano_coords.shape[0]),
            updatable_cov)

    # Loop over measurement chunks.
    for i, (F_part, data_part) in enumerate(zip(
                torch.chunk(F, chunks=20, dim=0),
                torch.chunk(data_values, chunks=20, dim=0))):
        print("Processing data chunk nr {}.".format(i))
        updatable_cov.update(F_part, data_std)
        updatable_mean.update(data_part, F_part)
        m = updatable_mean.m.cpu().numpy()
        np.save("m_post_{}.npy".format(i), m)

        # Save to VTK format..
        # Have to put back in rectangular grid.
        m_post_reg = np.zeros(reg_coords.shape[0])
        m_post_reg[volcano_inds] = m
        save_vtk(m_post_reg, (nx, ny, nz), res_x, res_y, res_z,
                "m_post_{}.mhd".format(i))


    """
    # Check that the covariance is correct by computing its product with a
    # dummy matrix.
    test_matrix = torch.rand(F_full.shape[1], 2000)
    res_test = updatable_cov.mul_right(test_matrix)
    """

if __name__ == "__main__":
    main()
