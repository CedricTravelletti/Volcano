# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Script running the inversion on synthetic dataset.
"""
from volcapy.inverse.gaussian_process import GaussianProcess
# import volcapy.covariance.covariance_tools as cl
import volcapy.covariance.matern32 as cl

import numpy as np
import os


# Should refactor this.
# This had to be inserted to make the sript run-protected for autodoc.
def prelude():
    # Should be loaded from metadata file.
    nx = 80
    ny = 80
    nz = 80
    res_x = 50
    res_y = 50
    res_z = 50
    
    
    # Set up logging.
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Now torch in da place.
    import torch
    
    # General torch settings and devices.
    torch.set_num_threads(8)
    gpu = torch.device('cuda:0')
    cpu = torch.device('cpu')
    
    # ----------------------------------------------------------------------------#
    #      LOAD DATA
    # ----------------------------------------------------------------------------#
    data_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/out/"
    # data_folder = "/idiap/temp/ctravelletti/tflow/Volcano/volcapy/synthetic/out"
    
    # Regular grid.
    reg_coords = np.load(os.path.join(data_folder, "reg_coords_synth.npy"))
    volcano_inds = np.load(os.path.join(data_folder, "volcano_inds_synth.npy"))
    data_values = np.load(os.path.join(data_folder, "data_values_synth.npy"))
    F = np.load(os.path.join(data_folder, "F_synth.npy"))
    
    n_data = data_values.shape[0]
    
    # Careful: we have to make a column vector here.
    data_std = 0.1
    
    d_obs = data_values.astype(np.float32)
    
    # Indices of the volcano inside the regular grid.
    volcano_coords = reg_coords.astype(np.float32)[volcano_inds]
    F = F.astype(np.float32)
    
    d_obs = torch.as_tensor(data_values[:, None]).float()
    volcano_coords = torch.as_tensor(volcano_coords).detach().float()
    F = torch.as_tensor(F).float()
    
    data_cov = torch.eye(n_data, dtype=torch.float32)
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    
    # ----------------------------------------------------------------------------#
    #     HYPERPARAMETERS
    # ----------------------------------------------------------------------------#
    sigma0_init = 193.85703
    m0 = 1439.846
    lambda0 = 422.0
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    
    ###########
    # IMPORTANT
    ###########
    out_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/forwards"
    # out_folder = "/idiap/temp/ctravelletti/tflow/Volcano/volcapy/synthetic/forwards"
    
    # Create the GP model.
    data_std = 0.1
    myGP = GaussianProcess(F, d_obs, data_cov, sigma0_init,
            data_std)


def main(out_folder, lambda0, sigma0):
    # Run prelude.
    prelude()

    # Create the covariance pushforward.
    cov_pushfwd = cl.compute_cov_pushforward(
            lambda0, F, volcano_coords, cpu, n_chunks=200,
            n_flush=50)
    K_d = torch.mm(F, cov_pushfwd)

    """
    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True)
    """
    m_post_d = myGP.condition_data(
            K_d, sigma0, concentrate=True)

    # Compute diagonal of posterior covariance.
    post_cov_diag = myGP.compute_post_cov_diag(
            cov_pushfwd, volcano_coords, lambda0, sigma0, cl)

    # Compute train_error
    train_error = myGP.train_RMSE()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True)

    # Compute train_error
    train_error = myGP.train_RMSE()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Save
    filename = "m_post_" + str(int(lambda0)) + "_sqexp.npy"
    np.save(os.path.join(out_folder, filename), m_post_m)

    filename = "post_cov_diag_" + str(int(lambda0)) + "_sqexp.npy"
    np.save(os.path.join(out_folder, filename), post_cov_diag)

    filename = "cov_pushfwd_" + str(int(lambda0)) + "_sqexp.npy"
    np.save(os.path.join(out_folder, filename), cov_pushfwd)

    # ---------------------------------------------
    # A AMELIORER
    # ---------------------------------------------
    # Save to VTK format..
    from volcapy.synthetic.vtkutils import save_vtk

    # Have to put back in rectangular grid.
    m_post_reg = np.zeros(reg_coords.shape[0])
    m_post_reg[volcano_inds] = m_post_m.numpy().reshape(-1)
    save_vtk(m_post_reg, (nx, ny, nz), res_x, res_y, res_z,
            "reconstructed_density.mhd")

if __name__ == "__main__":
    main(out_folder, lambda0, sigma0_init)
