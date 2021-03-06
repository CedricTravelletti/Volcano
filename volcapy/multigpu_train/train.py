# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.covariance.matern32 as cl

import gpytorch
import gpytorch.lazy

import numpy as np
import os


def main():
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
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    # niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
    # niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)
    n_data = inverseProblem.n_data
    
    
    # -- Delete Regularization Cells --
    # Delete the cells.
    # reg_cells_inds = get_regularization_cells_inds(inverseProblem)
    # inverseProblem.forward[:, reg_cells_inds] = 0.0
    
    F = torch.as_tensor(inverseProblem.forward).detach()
    
    # Careful: we have to make a column vector here.
    data_std = 0.1
    d_obs = torch.as_tensor(inverseProblem.data_values[:, None])
    data_cov = torch.eye(n_data)
    cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
    del(inverseProblem)

    # ----------------------------------------------------------------------------#
    # MULTIGPU STUFF
    # ----------------------------------------------------------------------------#
    output_device = torch.device('cuda:0')

    # make continguous
    cells_coords = cells_coords.contiguous()
    cells_coords_multigpu = cells_coords.to(output_device)
    
    F = F.contiguous()
    F_trans = F.t()
    F_trans_multigpu = F_trans.to(output_device)

    n_devices = torch.cuda.device_count()
    print('Planning to run on {} GPUs.'.format(n_devices))
    # base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    base_covar_module = gpytorch.kernels.RBFKernel()
    covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_covar_module, device_ids=range(n_devices),
                output_device=output_device)
    
    my_chunked_kernel = gpytorch.lazy.LazyEvaluatedKernelTensor(cells_coords,
            cells_coords, covar_module).to(output_device)
    
    checkpoint_size = 20000

    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    
    # ----------------------------------------------------------------------------#
    #     HYPERPARAMETERS
    # ----------------------------------------------------------------------------#
    sigma0_init = 500.0
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    
    ###########
    # IMPORTANT
    ###########
    out_folder = "/idiap/temp/ctravelletti/out/train/"
    
    # ---------------------------------------------------
    # Train multiple lambdas
    # ---------------------------------------------------
    # Range for the grid search.
    lambda0_start = 2.0
    lambda0_stop = 1400.0
    lambda0_step = 20.0
    lambda0s = np.arange(lambda0_start, lambda0_stop + 0.1, lambda0_step)
    n_lambda0s = len(lambda0s)
    logger.info("Number of lambda0s: {}".format(n_lambda0s))
    
    # Arrays to save the results.
    lls = np.zeros((n_lambda0s), dtype=np.float32)
    train_rmses = np.zeros((n_lambda0s), dtype=np.float32)
    loocv_rmses = np.zeros((n_lambda0s), dtype=np.float32)
    m0s = np.zeros((n_lambda0s), dtype=np.float32)
    sigma0s = np.zeros((n_lambda0s), dtype=np.float32)
    
    
    # OPTIMIZER LOGIC
    # The first lambda0 will be trained longer (that is, for the gradient descent
    # on sigma0). The next lambda0s will have optimal sigma0s that vary
    # continouslty, hence we can initialize with the last optimal sigma0 and train
    # for a shorter time.
    n_epochs_short = 6000
    n_epochs_long = 20000
    
    # Run gradient descent for every lambda0.
    from timeit import default_timer as timer
    start = timer()
    
    # Create the GP model.
    myGP = GaussianProcess(F, d_obs, sigma0_init,
            data_std=data_std, logger=logger)
    myGP.cuda()
    
    for i, lambda0 in enumerate(lambda0s):
        logger.info("Current lambda0 {} , {} / {}".format(lambda0, i, n_lambda0s))
    
        """
        # Compute the compute_covariance_pushforward and data-side covariance matrix
        cov_pushfwd = cl.compute_cov_pushforward(
                lambda0, F, cells_coords, gpu, n_chunks=200,
                n_flush=50)
        """
        my_chunked_kernel.kernel.base_kernel.lengthscale = lambda0
        # Compute the compute_covariance_pushforward and data-side covariance matrix
        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
            cov_pushfwd = my_chunked_kernel._matmul(F_trans)

        cov_pushfwd = cov_pushfwd.to(cpu)
        K_d = torch.mm(F, cov_pushfwd)
        
        # Perform the first training in full.
        # For the subsequent one, we can initialize sigma0 with the final value
        # from last training, since the optimum varies continuously in lambda0.
        # Hence, subsequent trainings can be shorter.
        if i > 0:
            n_epochs = n_epochs_short
        else: n_epochs = n_epochs_long
    
        # Run gradient descent.
        myGP.optimize(K_d, n_epochs, gpu, sigma0_init=None, lr=0.4)
    
        # Send everything back to cpu.
        myGP.to_device(cpu)
            
        # Once finished, run a forward pass.
        m_post_d = myGP.condition_data(K_d, sigma0=myGP.sigma0, concentrate=True)
        train_RMSE = myGP.train_RMSE()
        ll = myGP.neg_log_likelihood()
    
        # Compute LOOCV RMSE.
        loocv_rmse = myGP.loo_error()
    
        # Save the final ll, train/test error and hyperparams for each lambda.
        lls[i] = ll.item()
        train_rmses[i] = train_RMSE.item()
        loocv_rmses[i] = loocv_rmse.item()
        m0s[i] = myGP.m0
        sigma0s[i] = myGP.sigma0.item()
    
        # Save results every 5 iterations.
        if i % 4 == 0:
            logger.info("Saving at lambda0 {} , {} / {}".format(lambda0, i, n_lambda0s))
            np.save(os.path.join(out_folder, "log_likelihoods_train.npy"), lls)
            np.save(os.path.join(out_folder, "train_rmses_train.npy"), train_rmses)
            np.save(os.path.join(out_folder, "loocv_rmses_train.npy"), loocv_rmses)
            np.save(os.path.join(out_folder, "m0s_train.npy"), m0s)
            np.save(os.path.join(out_folder, "sigma0s_train.npy"), sigma0s)
            np.save(os.path.join(out_folder, "lambda0s_train.npy"), lambda0s)
    
    logger.info("Elapsed time:")
    end = timer()
    logger.info(end - start)
    # When everything done, save everything.
    logger.info("Finished. Saving results")
    np.save(os.path.join(out_folder, "log_likelihoods_train.npy"), lls)
    np.save(os.path.join(out_folder, "train_rmses_train.npy"), train_rmses)
    np.save(os.path.join(out_folder, "loocv_rmses_train.npy"), loocv_rmses)
    np.save(os.path.join(out_folder, "m0s_train.npy"), m0s)
    np.save(os.path.join(out_folder, "sigma0s_train.npy"), sigma0s)
    np.save(os.path.join(out_folder, "lambda0s_train.npy"), lambda0s)
    
    
    # Print optimal parameters.
    ind_min = np.argmin(lls)
    logger.info("OPTIMAL PARAMETERS")
    logger.info("------------------")
    logger.info("lambda0 {} , sigma0 {} , m0 {}".format(
            lambda0s[ind_min], sigma0s[ind_min], m0s[ind_min]))
    logger.info("Performance Metrics: Train RMSE {} , LOOCV RMSE {} , log-likelihood {}.".format(
            train_rmses[ind_min], loocv_rmses[ind_min], lls[ind_min]))

if __name__ == "__main__":
    main()
