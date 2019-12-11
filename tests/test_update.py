""" Test the updatable covariance module.

The first function just verifiies that everything runs. The second ones
verifies that it is correct by comparing direct conditioning on the whole
dataset with two steps conditioning.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.covariance.exponential as cl

import numpy as np
import os


def main():
    # Now torch in da place.
    import torch
    
    # General torch settings and devices.
    torch.set_num_threads(8)
    gpu = torch.device('cuda:0')
    cpu = torch.device('cpu')
    
    # ----------------------------------------------------------------------------#
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    # niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    # niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
    niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)

    # -- Delete Regularization Cells --
    # Delete the cells.
    reg_cells_inds = get_regularization_cells_inds(inverseProblem)
    inverseProblem.forward[:, reg_cells_inds] = 0.0

    # WARNING: GET RID OF FIRST LINE OF F, SINCE IT seems different.
    # Save full conditining matrix, before splitting the dataset.
    F_full = torch.as_tensor(inverseProblem.forward).detach()
    d_obs_full = torch.as_tensor(inverseProblem.data_values[:, None])
    
    # Test-Train split.
    n_keep = 300
    rest_forward, rest_data = inverseProblem.subset_data(n_keep, seed=2)
    n_data = inverseProblem.n_data
    F_test = torch.as_tensor(rest_forward).detach()
    d_obs_test = torch.as_tensor(rest_data[:, None]).detach()
    
    F = torch.as_tensor(inverseProblem.forward).detach()
    
    # Careful: we have to make a column vector here.
    data_std = 0.1
    d_obs = torch.as_tensor(inverseProblem.data_values[:, None])
    data_cov = torch.eye(n_data)
    cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
    del(inverseProblem)

    lambda0 = 100.0
    sigma0 = 1.0
    epsilon0 = 10.0

    # Now ready to go to updatable covariance.
    from volcapy.update.updatable_covariance import UpdatableCovariance
    updatable_cov = UpdatableCovariance(cl, lambda0, sigma0, epsilon0, cells_coords)

    # First conditioning.
    updatable_cov.update(F)

    # Second conditioning.
    updatable_cov.update(F_test)

    # Check that the covariance is correct by computing its product with a
    # dummy matrix.
    test_matrix = torch.rand(F.shape[1], 2000)
    res_test = updatable_cov.mul_right(test_matrix)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Compare with the *true* product (the one produced by conditioning
    # everything in one go).
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Create the GP model.
    myGP = GaussianProcess(F_full, d_obs_full, sigma0,
            data_std=data_std)
    myGP.cuda()

    true_pushfwd = cl.compute_cov_pushforward(lambda0, F_full, cells_coords,
            device=gpu,
            n_chunks=200, n_flush=50)
    K_d = torch.mm(F_full, true_pushfwd)
    m_post_d = myGP.condition_data(K_d, sigma0=myGP.sigma0, concentrate=True)

    # Compute the test product.
    # It involves the product of the matrix with C0 so a kind of *pushforward*.
    test_matrix_pushfwd = cl.compute_cov_pushforward(lambda0, test_matrix.t(), cells_coords,
            device=gpu,
            n_chunks=200, n_flush=50)
    res_true = sigma0**2 * test_matrix_pushfwd + torch.mm(
            torch.mm(true_pushfwd, myGP.inversion_operator),
            torch.mm(true_pushfwd.t(), test_matrix))

    print(res_true - res_test)
    print(res_true.shape)
    print(res_true)
    print(res_test)


if __name__ == "__main__":
    main()
