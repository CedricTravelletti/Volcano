# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Test the HEAVY refactor of gaussian process (06.02.2020).

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds

import numpy as np
import torch
import os


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
    
    G = np.delete(inverseProblem.forward, inds_to_delete, axis=1)
    cells_coords = np.delete(inverseProblem.cells_coords, inds_to_delete, axis=0)
    
    G = torch.as_tensor(G).detach()
    G = G.to(gpu0)
    cells_coords = torch.as_tensor(cells_coords).detach()

    data_std = 0.1
    y = torch.as_tensor(inverseProblem.data_values)
    y = y.reshape(y.shape[0], 1).to(gpu0)
    del(inverseProblem)
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#

    # Hyperparams.
    m0, sigma0, lambda0 = 1800.0, 400.0, 200.0
    
    # Create the GP model.
    import volcapy.covariance.matern32 as kernel
    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            cells_coords, kernel,
            logger=logger)
    myGP.cuda() # See if still necessary.
            
    # Run a forward pass.
    m_post_d = myGP.condition_data(G, y, data_std, concentrate=False)

    # Try model conitioning.
    m_post_m, m_post_d = myGP.condition_model(G, y, data_std, concentrate=False,
            is_precomp_pushfwd=True)

    # Same, but this time do not re-use pushforward.
    m_post_m, m_post_d = myGP.condition_model(G, y, data_std, concentrate=False)

    myGP.train_fixed_lambda(lambda0, G, y, data_std,
            n_epochs=5000, lr=0.05)
         

if __name__ == "__main__":
    main()
