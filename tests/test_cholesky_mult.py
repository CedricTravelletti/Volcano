""" Test the routine that implements multiplication of inversion operator with
another test matrix (using Cholesk triangular solves.

Also check how direct multiplication with the inverse (computed with Cholesky)
and multiplication with triangular solve compare as a function of the condition
number.

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
    reg_cells_inds, bottom_inds = get_regularization_cells_inds(inverseProblem)
    inds_to_delete = list(set(
        np.concatenate([reg_cells_inds, bottom_inds], axis=0)))
    
    F = np.delete(inverseProblem.forward, inds_to_delete, axis=1)
    cells_coords = np.delete(inverseProblem.cells_coords, inds_to_delete, axis=0)
    
    F_full = torch.as_tensor(F).detach()
    cells_coords = torch.as_tensor(cells_coords).detach()
    
    d_obs_full = torch.as_tensor(inverseProblem.data_values[:, None])
    del(inverseProblem)
    
    # Test-Train split.
    n_keep = 300
    F_part_1 = F_full[:n_keep, :]
    F_part_2 = F_full[n_keep:, :]

    # rest_forward, rest_data = inverseProblem.subset_data(n_keep, seed=2)
    
    
    # Careful: we have to make a column vector here.
    data_std = 0.1
    lambda0 = 100.0
    sigma0 = 1.0

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

    # Now try to multiply the inversion operator with a matrix and see what we
    # get.
    triangular_solve_res = myGP.inv_op_vector_mult(myGP.R)
    cholesky_inverse_res = torch.mm(myGP.inversion_operator, myGP.R)

    print("Result with triangular solve.")
    print(triangular_solve_res)

    print("Result with Cholesky inversion.")
    print(cholesky_inverse_res)

if __name__ == "__main__":
    main()
