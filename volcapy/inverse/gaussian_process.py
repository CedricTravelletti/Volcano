""" Gaussian Process for Inversion class.

This class implements gaussian process regression/conditioning for inverse
problems, i.e. when conditioning on a linear operator of the field.

The situation is we have a gaussian process on some space and a linear
operator, denoted F, acting on the process and mapping it to another space.

This linear operator induces a gaussian process on the target space.

We will use the term *model* when we refer to the original gaussian process and
*data* when we refer to the induced gaussian process.

We discretize model-space and data-space, so that F becomes a matrix.

Notation and Implementation
---------------------------
We denote by K the covariance matrix of the original field. We assume a
constant prior mean vector mu0 = m0 * 1_m, where 1_m stands for the identity
vector in model space. Hence we only have one scalar parameter m0 for the prior mean.

The induced gaussian process then has covariance matrix K_d = F K F^t and
prior mean vector m0 * F * 1_m.

Regression/Conditioning
-----------------------
We observe the value of the induced field at some points in data-space. We then
condition our GP on those observations and obtain posterior mean vector /
covariance matrix.

Covariance Matrices
-------------------
Most kernels have a variance parameter sigma0^2 that just appears as a
multiplicative constant. To make its optimization easier, we strip it of the
covariance matrix, store it as a model parameter (i.e. as an attribute of the
class) and include it manually in the experessions where it shows up.

This means that when one sees a covariance matrix in the code, it generally
doesn't include the sigma0 factor, which has to be included by hand.


The model covariance matrix K is to big to be stored, but during conditioning
it only shows up in the form K * F^t. It is thus sufficient to compute this
product once and for all. We call it the *covariance pushforward*.

"""
import numpy as np
import torch
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


# Noise to Variance ratio.
# NtV = (0.001)**2

# Data standard deviation.
epsilon = 0.1


class GaussianProcess(torch.nn.Module):
    def __init__(self, F, d_obs, data_ones, sigma0_init, data_std=0.1):
        """

        Parameters
        ----------
        F
            Forward operator matrix
        d_obs
            Observed data vector.
        data_ones
            Data (observations) covariance matrix.
        sigma0_init
            Original value of the sigma0 parameter to use when starting
            optimization.

        """
        super(GaussianProcess, self).__init__()
        # Store the sigma0 after optimization, since can be used as starting
        # point for next optim.
        self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0_init))

        # Sizes
        self.n_model = F.shape[1]
        self.n_data = F.shape[0]

        # Prior mean (vector) on the data side.
        self.mu0_d_stripped = torch.mm(F, torch.ones((self.n_model, 1),
                dtype=torch.float32))

        self.d_obs = d_obs
        self.data_ones = torch.eye(self.n_data, dtype=torch.float32)
        self.data_std = data_std

        # Identity vector. Need for concentration.
        self.I_d = torch.ones((self.n_data, 1), dtype=torch.float32)

    def to_device(self, device):
        """ Transfer all model attributes to the given device.
        Can be used to switch between cpu and gpu.

        Parameters
        ----------
        device: torch.Device

        """
        self.sigma0 = torch.nn.Parameter(self.sigma0.to(device))
        self.mu0_d_stripped = self.mu0_d_stripped.to(device)
        self.d_obs = self.d_obs.to(device)
        self.data_ones = self.data_ones.to(device)
        self.I_d = self.I_d.to(device)

    def neg_log_likelihood(self):
        """ Computes the negative log-likelihood of the current state of the
        model.
        Note that this function should be called AFTER having run a
        conditioning, since it depends on the inversion operator computed
        there.

        Returns
        -------
        float

        """
        # Need to do it this way, otherwise rounding errors kill everything.
        """
        log_det = (
                - torch.logdet(self.stripped_inv)
                + 2 * torch.log(self.sigma0))
        """
        # Note that for some reason, the above doesnt work the same.
        # We should really check the implementation of logdet.
        """
        log_det = - torch.logdet(self.inversion_operator)
        """
        log_det = -2*torch.log(self.sigma0) - torch.logdet(self.stripped_inv)

        nll = torch.add(
                log_det,
                (1 / self.sigma0**2) * torch.mm(
                      self.prior_misfit.t(),
                      torch.mm(self.stripped_inv, self.prior_misfit)))
        return nll

    def concentrate_m0(self):
        """ Compute m0 (prior mean parameter) by MLE via concentration.

        Note that the inversion operator should have been updated first.

        """
        conc_m0 = torch.mm(
                torch.inverse(
                    torch.mm(
                        torch.mm(self.mu0_d_stripped.t(), self.stripped_inv),
                        self.mu0_d_stripped)),
                torch.mm(
                    self.mu0_d_stripped.t(),
                    torch.mm(self.stripped_inv, self.d_obs)))

        return conc_m0

    def condition_data(self, K_d, sigma0, m0=0.1, concentrate=False,
            NtV_crit=-1.0):
        """ Condition model on the data side.

        Parameters
        ----------
        K_d
            Covariance matrix on the data side.
        sigma0
            Standard deviation parameter for the kernel.
        m0
            Prior mean parameter.
        concentrate
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood.
        NtV_crit:
            Critical Noise to Variance ratio (epsilon/sigma0). We can fix it to avoid
            numerical instability in the matrix inversion.
            If provided, will not allow to go below the critical value.

        Returns
        -------
        mu_post_d
            Posterior mean data vector

        """
        # Noise to Variance ratio.
        # If not specified, then do not modify it.
        NtV = (self.data_std / sigma0)**2
        if NtV < NtV_crit:
            NtV = NtV_crit

        inv_inversion_operator = torch.add(
                        NtV * self.data_ones, K_d)

        # Store for the logdet. Should refactor later.
        self.stripped_inv = torch.inverse(inv_inversion_operator)

        # Compute inversion operator and store once and for all.
        self.inversion_operator = (
                (1 / sigma0**2) * self.stripped_inv)

        if concentrate:
            # Determine m0 (on the model side) from sigma0 by concentration of the Ll.
            m0 = self.concentrate_m0()

        self.mu0_d = m0 * self.mu0_d_stripped
        # Store m0 in case we want to print it later.
        self.m0 = m0
        self. prior_misfit = torch.sub(self.d_obs, self.mu0_d)
        weights = torch.mm(self.inversion_operator, self.prior_misfit)

        mu_post_d = torch.add(
                self.mu0_d,
                torch.mm(sigma0**2 * K_d, weights))
        # Store in case.
        self.mu_post_d = mu_post_d

        return mu_post_d

    def condition_model(self, cov_pushfwd, F, sigma0, m0=0.1,
            concentrate=False, NtV_crit=-1.0):
        """ Condition model on the model side.

        Parameters
        ----------
        cov_pushfwd
            Pushforwarded covariance matrix, i.e. K F^T
        F
            Forward operator
        sigma0
            Standard deviation parameter for the kernel.
        m0
            Prior mean parameter.
        concentrate
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood.
        NtV_crit:
            Critical Noise to Variance ratio (epsilon/sigma0). We can fix it to avoid
            numerical instability in the matrix inversion.
            If provided, will not allow to go below the critical value.

        Returns
        -------
        mu_post_m
            Posterior mean model vector
        mu_post_d
            Posterior mean data vector

        """
        # Noise to Variance ratio.
        # If not specified, then do not modify it.
        NtV = (self.data_std / sigma0)**2
        if NtV < NtV_crit:
            NtV = NtV_crit

        inv_inversion_operator = torch.add(
                        NtV * self.data_ones,
                        torch.mm(F, cov_pushfwd))

        # Store for the logdet. Should refactor later.
        self.stripped_inv = torch.inverse(inv_inversion_operator)

        # Compute inversion operator and store once and for all.
        self.inversion_operator = (
                (1 / sigma0**2) * self.stripped_inv)

        if concentrate:
            # Determine m0 (on the model side) from sigma0 by concentration of the Ll.
            m0 = self.concentrate_m0()

        # Prior mean for data and model.
        self.mu0_d = m0 * self.mu0_d_stripped
        self.mu0_m = m0 * torch.ones((self.n_model, 1))

        # Store m0 in case we want to print it later.
        self.m0 = m0
        self. prior_misfit = torch.sub(self.d_obs, self.mu0_d)
        weights = torch.mm(self.inversion_operator, self.prior_misfit)

        # -----------------------------------------
        # -----------------------------------------
        # WARNING
        # We try to compute inverse matrix - vector product using Cholesky.
        # See if its better.
        # -----------------------------------------
        # -----------------------------------------
        u = torch.cholesky(inv_inversion_operator)
        z2, _ = torch.triangular_solve(self.prior_misfit, u, upper=False)
        y2, _ = torch.triangular_solve(z2, u.T, upper=True)
        weights = (1 / sigma0**2) * y2





        # Posterior data mean.
        self.mu_post_d = torch.add(
                self.mu0_d,
                torch.mm(sigma0**2 * torch.mm(F, cov_pushfwd), weights))

        # Posterior model mean.
        self.mu_post_m = torch.add(
                self.mu0_m,
                torch.mm(sigma0**2 * cov_pushfwd, weights))

        return self.mu_post_m.detach(), self.mu_post_d

    def optimize(self, K_d, n_epochs, device, logger, sigma0_init=None,
            lr=0.007, NtV_crit=-1.0):
        """ Given lambda0, optimize the two remaining hyperparams via MLE.
        Here, instead of giving lambda0, we give a (stripped) covariance
        matrix. Stripped means without sigma0.

        The user can choose between CPU and GPU.

        Parameters
        ----------
        K_d: 2D Tensor
            (stripped) Covariance matrix in data space.
        n_epochs: int
            Number of training epochs.
        device: Torch.device
            Device to use for optimization, either CPU or GPU.
        logger
            An instance of logging.Logger, used to output training progression.
        sigma0_init: float
            Starting value for gradient descent. If None, then use the value
            sotred by the model class (that is, the one resulting from the
            previous optimization run).
        lr: float
            Learning rate.

        """
        # Send everything to the correct device.
        self.to_device(device)
        K_d = K_d.to(device)

        # Initialize sigma0.
        if sigma0_init is not None:
            self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0_init)).to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            # Forward pass: Compute predicted y by passing
            # x to the model
            m_posterior_d = self.condition_data(K_d, self.sigma0,
                    concentrate=True,
                    NtV_crit=NtV_crit)
            log_likelihood = self.neg_log_likelihood()

            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            log_likelihood.backward(retain_graph=True)
            optimizer.step()

            # Periodically print informations.
            if epoch % 100 == 0:
                # Compute train error.
                train_RMSE = self.train_RMSE()
                logger.info("sigma0: {}".format(self.sigma0.item()))
                logger.info("Log-likelihood: {}".format(log_likelihood.item()))
                logger.info("RMSE train error: {}".format(train_RMSE.item()))

        logger.info("Log-likelihood: {}".format(log_likelihood.item()))
        logger.info("RMSE train error: {}".format(train_RMSE.item()))

        # Send everything back to cpu.
        self.to_device(cpu)

        return

    def post_cov(self, cov_pushfwd, cells_coords, lambda0, sigma0, i, j, cl):
        """ Condition model on the model side.

        Parameters
        ----------
        cov_pushfwd
            Pushforwarded covariance matrix, i.e. K F^T
        cells_coords: tensor
            n_cells * n_dims: cells coordinates
        lambda0: float
            Lenght-scale parameter
        sigma0: float
            Prior variance.
        i: int
            Index of first cell (index in the cells_coords array).
        j: int
            Index of second cell.

        Returns
        -------
        post_cov

        """
        cov = cl.compute_cov(lambda0, cells_coords, i, j)
        post_cov = sigma0**2 * (cov -
                torch.mm(
                    sigma0**2 * cov_pushfwd[i, :].reshape(1, -1),
                    torch.mm(
                        self.inversion_operator, cov_pushfwd[j, :].reshape(-1, 1))))

        return post_cov

    def compute_post_cov_diag(self, cov_pushfwd, cells_coords, lambda0, sigma0,
            cl):
        n_cells = cells_coords.shape[0]
        post_cov_diag = np.zeros(n_cells)

        for i in range(n_cells):
            post_cov_diag[i] = self.post_cov(
                    cov_pushfwd, cells_coords, lambda0, sigma0, i, i, cl)

        return post_cov_diag

    def loo_predict(self, loo_ind):
        """ Leave one out krigging prediction.

        Take the trained hyperparameters. Remove one point from the
        training set, krig/condition on the remaining point and predict
        the left out point.

        WARNING: Should have run the forward pass of the model once before,
        otherwise some of the operators we need (inversion operator) won't have
        been computed. Also should re-run the forward pass when updating
        hyperparameters.

        Parameters
        ----------
        loo_ind: int
            Index (in the training set) of the data point to leave out.

        Returns
        -------
        float
            Prediction at left out data point.

        """
        # Index of the not removed data points.
        in_inds = list(range(len(self.d_obs)))
        in_inds.remove(loo_ind)

        # Note that for the dot product, we should have one-dimensional
        # vectors, hence the strange indexing with the zero.
        loo_pred = (self.mu0_d[loo_ind] -
                1/self.inversion_operator[loo_ind, loo_ind].detach() *
                torch.dot(
                    self.inversion_operator[loo_ind, in_inds].detach(),
                    self.prior_misfit[in_inds, 0].detach()))

        return loo_pred.detach()

    def loo_error(self):
        """ Leave one out cross validation RMSE.

        Take the trained hyperparameters. Remove one point from the
        training set, krig/condition on the remaining point and predict
        the left out point.
        Compute the squared error, repeat for all data points (leaving one out
        at a time) and average.

        WARNING: Should have run the forward pass of the model once before,
        otherwise some of the operators we need (inversion operator) won't have
        been computed. Also should re-run the forward pass when updating
        hyperparameters.

        Returns
        -------
        float
            RMSE cross-validation error.

        """
        tot_error = 0
        for loo_ind in range(len(self.d_obs)):
            loo_prediction = self.loo_predict(loo_ind)
            tot_error += (self.d_obs[loo_ind].item() - loo_prediction)**2

        return torch.sqrt((tot_error / len(self.d_obs)))

    def train_RMSE(self):
        """ Compute current error on training set.

        """
        return torch.sqrt(torch.mean(
            (self.d_obs - self.mu_post_d)**2))

    def condition_number(self, cov_pushfwd, F, sigma0):
        """ Compute condition number of the inversion matrix R.

        Used to detect numerical instability.

        Parameters
        ----------
        cov_pushfwd
            Pushforwarded covariance matrix, i.e. K F^T
        F
            Forward operator
        sigma0
            Standard deviation parameter for the kernel.

        Returns
        -------
        float
            Condition number of the inversion matrix R.

        """
        # Noise to Variance ratio.
        NtV = (self.data_std / sigma0)**2

        inv_inversion_operator = torch.add(
                        NtV * self.data_ones,
                        torch.mm(F, cov_pushfwd))

        return np.linalg.cond(inv_inversion_operator.detach().numpy())

    def subset_data(self, n_keep):
        """ Subset an inverse problem by only keeping n_keep data
        points selected at random.

        The effect of this method is to directly modify the attributes of the
        class to adapt them to the smaller problem.

        Parameters
        ----------
        n_keep: int
            Only keep n_keep data points.

        Returns
        -------
        (rest_forward, rest_data)


        """
        # Pick indices at random.
        inds = np.random.choice(range(self.n_data), n_keep, replace=False)

        # Return the unused part of the forward and of the data so that it can
        # be used as test set.
        rest_forward = np.delete(self.forward, inds, axis=0)
        rest_data = np.delete(self.data_values, inds, axis=0)
        rest_data_points = np.delete(self.data_points, inds, axis=0)

        # Now subset
        self.forward = self.forward[inds, :]
        self.data_points = self.data_points[inds, :]
        self.data_values = self.data_values[inds]
        # We subsetted columns, hence we have to restore C-contiguity by hand.
        self.forward = np.ascontiguousarray(self.forward)

        # New dimensions
        self.n_data = self.forward.shape[0]

        # Note that we return data in a column vector, to make it
        # compatible with the rest of the data.
        return (rest_forward, rest_data[:, None])
