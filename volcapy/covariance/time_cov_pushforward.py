""" This script is meant to time the computation of the covariance pushforward
using our original implementation to see if its worth switching to KeOps.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.covariance.squared_exponential as cl

import torch
import numpy as np

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


def main():
    # With PyTorch, using the GPU is that simple:
    # ----------------------------------------------------------------------------#
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    # niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
    # niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)

    F = torch.as_tensor(inverseProblem.forward).detach()
    cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
    del(inverseProblem)

    lambda0 = 100.0

    # Time the execution to know if its worth switching to KeOps.
    from timeit import default_timer as timer
    start = timer()

    cov_pushfwd = cl.compute_cov_pushforward(
            lambda0, F, cells_coords, gpu, n_chunks=200,
            n_flush=50)

    end = timer()
    print((end - start)/60.0)

    """
    v    = K_ij@p  # Genuine torch Tensor. (N,N)@(N,D) = (N,D)

    # Finally, compute the kernel norm H(q,p):
    H = .5 * torch.dot( p.view(-1), v.view(-1) ) # .5 * <p,v>

    """

if __name__ == "__main__":
    main()
