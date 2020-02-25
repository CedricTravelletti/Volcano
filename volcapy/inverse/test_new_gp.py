# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Test the HEAVY refactor of gaussian process (06.02.2020).

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds

import numpy as np
import torch
import os

torch.set_num_threads(1)

out_path = "./train_results.pkl"


def main():
    # Get GPUs.
    gpu0 = torch.device('cuda:0')

    # Set up logging.
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # ----------------------------------------------------------------------------#
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    # niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    # niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
    niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)
    
    # -- Delete Regularization Cells --
    reg_cells_inds, bottom_inds = get_regularization_cells_inds(inverseProblem)
    inds_to_delete = list(set(
        np.concatenate([reg_cells_inds, bottom_inds], axis=0)))
    
    G = inverseProblem.forward
    cells_coords = inverseProblem.cells_coords

    """
    G = np.delete(G, inds_to_delete, axis=1)
    cells_coords = np.delete(cells_coords, inds_to_delete, axis=0)
    """
    
    G = torch.as_tensor(G)
    G = G.to(gpu0)
    cells_coords = torch.as_tensor(cells_coords).to(gpu0)

    data_std = 0.1
    y = torch.as_tensor(inverseProblem.data_values)
    y = y.reshape(y.shape[0], 1).to(gpu0)
    del(inverseProblem)
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#

    # Hyperparams.
    m0, sigma0, lambda0 = 1800.0, 775.0, 20.0
    
    # Create the GP model.
    import volcapy.covariance.matern32 as kernel
    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            cells_coords, kernel,
            logger=logger)
    myGP.cuda() # See if still necessary.
            
    """
    # Run a forward pass.
    m_post_d, nll, data_std = myGP.condition_data(G, y, data_std, concentrate=False)

    # Try model conitioning.
    m_post_m, m_post_d = myGP.condition_model(G, y, data_std, concentrate=False,
            is_precomp_pushfwd=True)

    # Same, but this time do not re-use pushforward.
    m_post_m, m_post_d = myGP.condition_model(G, y, data_std, concentrate=False)

    """

    lambda0_start, lambda0_stop, lambda0_step = 20, 700, 20
    lambda0s = np.arange(lambda0_start, lambda0_stop + 0.1, lambda0_step)

    myGP.train(lambda0s, G, y, data_std,
            out_path,
            n_epochs=3000, lr=0.2)
    

if __name__ == "__main__":
    main()
