# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Fit the hyperparameters of a gaussian process model to the synthetic
datasetgenerated by :code:`build_synth_data`.

"""
from volcapy.inverse.gaussian_process import GaussianProcess
import volcapy.covariance.covariance_tools as cl

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
    #      LOAD DATA
    # ----------------------------------------------------------------------------#
    # data_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/out/"
    data_folder = "/idiap/temp/ctravelletti/tflow/Volcano/volcapy/synthetic/synthetic_data"
    
    reg_cells_coords = np.load(os.path.join(data_folder, "coords_synth.npy"))
    volcano_inds = np.load(os.path.join(data_folder, "volcano_inds_synth.npy"))
    cells_coords = reg_cells_coords[volcano_inds]
    
    data_values = np.load(os.path.join(data_folder, "data_values_synth.npy"))
    F = np.load(os.path.join(data_folder, "F_synth.npy"))
    
    n_data = data_values.shape[0]
    
    d_obs = data_values.astype(np.float32)
    cells_coords = cells_coords.astype(np.float32)
    F = F.astype(np.float32)
    
    d_obs = torch.as_tensor(data_values[:, None]).float()
    cells_coords = torch.as_tensor(cells_coords).detach().float()
    F = torch.as_tensor(F).float()
    
    data_cov = torch.eye(n_data, dtype=torch.float32)
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    
    # ----------------------------------------------------------------------------#
    #     HYPERPARAMETERS
    # ----------------------------------------------------------------------------#
    sigma0_init = 200.0
    m0 = 2200.0
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    
    ###########
    # IMPORTANT
    ###########
    out_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/forwards"
    
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
    lambda0_stop = 600.0
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
    n_epochs_short = 4000
    n_epochs_long = 10000
    
    # Run gradient descent for every lambda0.
    from timeit import default_timer as timer
    start = timer()
    
    # Create the GP model.
    data_std = 1000.0
    myGP = GaussianProcess(F, d_obs, data_cov, sigma0_init,
            data_std)
    myGP.cuda()
    
    for i, lambda0 in enumerate(lambda0s):
        logger.info("Current lambda0 {} , {} / {}".format(lambda0, i, n_lambda0s))
    
        # Compute the compute_covariance_pushforward and data-side covariance matrix
        cov_pushfwd = cl.compute_cov_pushforward(
                lambda0, F, cells_coords, gpu, n_chunks=200,
                n_flush=50)
        K_d = torch.mm(F, cov_pushfwd)
        
        # Perform the first training in full.
        # For the subsequent one, we can initialize sigma0 with the final value
        # from last training, since the optimum varies continuously in lambda0.
        # Hence, subsequent trainings can be shorter.
        if i > 0:
            n_epochs = n_epochs_short
        else: n_epochs = n_epochs_long
    
        # Run gradient descent.
        myGP.optimize(K_d, n_epochs, gpu, logger, sigma0_init=None, lr=0.5)
    
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

if __name__ == "__main__":
    main()
