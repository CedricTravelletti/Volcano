" KeOps implementation of squared exponential covariance.

We here also try to take an object oriented approach to kernels.

IMPORTANT: As always, we strip the variance parameter (i.e. set sigma_0 = 1).

WARNING WARNING
WARNING WARNING: currently cells_coords is passed by reference. If we want to
provide an update_cells method, we need to make a copy, because otherwise we
will be modifying stuff outside our scope.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.covariance.matern32 as cl

import torch
import numpy as np

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


def main():
    # With PyTorch, using the GPU is that simple:
    use_gpu = torch.cuda.is_available()
    dtype   = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    # Under the hood, this flag determines the backend that is to be
    # used for forward and backward operations, which have all been
    # implemented both in pure CPU and GPU (CUDA) code.

    # ----------------------------------------------------------------------------#
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    # niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    # niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
    niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)

    cells_coords = torch.as_tensor(inverseProblem.cells_coords[:1000,:])
    del(inverseProblem)

    lambda0 = 100.0
    lambda0 = lambda0.type(dtype)
    cells_coords = cells_coords.type(dtype)

    cells_coords.requires_grad = True

    q_i  = LazyTensor(self.cells_coords[:,None,:])  # shape (N,D) -> (N,1,D)
    q_j  = LazyTensor(self.cells_coords[None,:,:])  # shape (N,D) -> (1,N,D)

    D_ij = ((q_i - q_j) ** 2).sum(dim=2)  # Symbolic matrix of squared distances
    K_ij = (- D_ij / (2 * self.lambda0**2) ).exp()   # Symbolic Gaussian kernel matrix

    v    = K_ij@p  # Genuine torch Tensor. (N,N)@(N,D) = (N,D)

    # Finally, compute the kernel norm H(q,p):
    H = .5 * torch.dot( p.view(-1), v.view(-1) ) # .5 * <p,v>

if __name__ == "__main__":
    main()
