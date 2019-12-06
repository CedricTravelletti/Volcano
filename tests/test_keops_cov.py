""" Test the KeOps implementation of our covariance modules.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.covariance.matern32 as cl

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
    # niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
    niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)
    
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

