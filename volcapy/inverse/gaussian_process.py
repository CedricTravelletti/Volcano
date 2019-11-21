"""
This class implements gaussian process regression/conditioning for inverse
problems, i.e. when conditioning on a linear operator of the field.
The situation is we have a gaussian process on some space and a linear
operator, denoted :math:`F`, acting on the process and mapping it to another space.
This linear operator induces a gaussian process on the target space.
We will use the term *model* when we refer to the original gaussian process and
*data* when we refer to the induced gaussian process.
We discretize model-space and data-space, so that F becomes a matrix.

Notation and Implementation
---------------------------
We denote by :math:`K` the covariance matrix of the original field. We assume a
constant prior mean vector :math:`\mu_0 = m_0 1_m`, where :math:`1_m` stands for the identity
vector in model space. Hence we only have one scalar parameter m0 for the prior mean.

The induced gaussian process then has covariance matrix :math:`K_d = F K F^t` and
prior mean vector :math:`m_0 F ~ 1_m`.

Regression/Conditioning
-----------------------
We observe the value of the induced field at some points in data-space. We then
condition our GP on those observations and obtain posterior mean vector /
covariance matrix.

Covariance Matrices
-------------------
Most kernels have a variance parameter :math:`\sigma_0^2` that just appears as a
multiplicative constant. To make its optimization easier, we strip it of the
covariance matrix, store it as a model parameter (i.e. as an attribute of the
class) and include it manually in the experessions where it shows up.

This means that when one sees a covariance matrix in the code, it generally
doesn't include the :math:`\sigma_0` factor, which has to be included by hand.

Covariance Pushforward
----------------------
The model covariance matrix :math:`K` is to big to be stored, but during conditioning
it only shows up in the form :math:`K F^t`. It is thus sufficient to compute this
product once and for all. We call it the *covariance pushforward*.

Noise
-----
Independent homoscedactic noise on the data is assumed. It is specified by
*data_std_orig*. If the matrices are ill-conditioned, the the noise will have
to be increased, hence the current value of the noise will be stored in
*noise_std*.

"""
import numpy as np
import torch
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


class GaussianProcess(torch.nn.Module):
    """

    Attributes
    ----------
    sigma0_init
        Original value of the sigma0 parameter to use when starting
        optimization.
    sigma0: float
        Current value of the prior standard deviation of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    data_std_orig: float
        Original data noise standard deviation.
    data_std: float
        Current data noise deviation. Can change from the original value
        since we may have to increase it to make matrices invertible/better
        conditioned.
    F: ndarray
        Forward operator matrix, n_d * n_m
    d_obs: tensor
       Vector of observed data values
    n_model: int
        Number of model cells.
    n_data: int
        Number of data observations.
    mu0_d_stripped: Tensor
        The pushforwarded mean vector, stripped of the constant prior means
        parameter. I.e. this is F * 1_m. Multiply by the prior mean m0 to get
        the data-side mean.
    data_ones
    inv_op_L: Tensor
        Lower Cholesky factor of the inverse inversion operator R.
    inversion_operator
    mu0_d
    m0
    prior_misfit
    weights
    mu_post_d
    mu_post_m
    stripped_inv
    mu0_m
    R: Tensor.
        Inverse inversion operator. Given by
        data_std**2 * 1_{d*d} + sigma0**2 * K_d

    logger
        An instance of logging.Logger, used to output training progression.

    """
    def __init__(self, F, d_obs, sigma0_init, data_std=0.1,
            logger=None):
        """

        Parameters
        ----------
        F: ndarray
            Forward operator matrix, n_d * n_m
        d_obs
            Observed data vector.
        sigma0_init
            Original value of the sigma0 parameter to use when starting
            optimization.
        data_std: float
            Data noise standard deviation.
        logger
            An instance of logging.Logger, used to output training progression.

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

        self.data_std_orig = data_std
        # Used to store the modified noise level, if has to be increased to
        # make matrix invertible.
        self.data_std = data_std

        # If no logger, create one.
        if logger is None:
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        self.logger = logger

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
        # WARNING!!! determinant is not linear! Taking constants outside adds
        # power to them.
        log_det = torch.logdet(self.R)
        nll = log_det + torch.mm(self.prior_misfit.t(), self.weights)

        return nll

    def concentrate_m0(self):
        """ Compute m0 (prior mean parameter) by MLE via concentration.

        Note that the inversion operator should have been updated first.

        """
        # Compute R^(-1) * G * I_m.
        tmp = self.inv_op_vector_mult(self.mu0_d_stripped)
        conc_m0 = torch.mm(self.d_obs.t(), tmp) / torch.mm(self.mu0_d_stripped.t(), tmp)

        return conc_m0

    def condition_data(self, K_d, sigma0, m0=0.1, concentrate=False):
        """ Condition model on the data side.

        Parameters
        ----------
        K_d: tensor
            Covariance matrix on the data side.
        sigma0: float
            Standard deviation parameter for the kernel.
        m0: float
            Prior mean parameter.
        concentrate: bool
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood.

        Returns
        -------
        mu_post_d: tensor
            Posterior mean data vector

        """
        # Get Cholesky factor (lower triangular) of the inversion operator.
        self.inv_op_L = self.get_inversion_op_cholesky(K_d, sigma0)
        self.inversion_operator = torch.cholesky_inverse(self.inv_op_L)
            
        if concentrate:
            # Determine m0 (on the model side) from sigma0 by concentration of the Ll.
            m0 = self.concentrate_m0()

        self.mu0_d = m0 * self.mu0_d_stripped
        # Store m0 in case we want to print it later.
        self.m0 = m0
        self. prior_misfit = self.d_obs - self.mu0_d

        self.weights = self.inv_op_vector_mult(self.prior_misfit)

        mu_post_d = self.mu0_d + torch.mm(sigma0**2 * K_d, self.weights)
        # Store in case.
        self.mu_post_d = mu_post_d

        return mu_post_d

    def condition_model(self, cov_pushfwd, F, sigma0, m0=0.1,
            concentrate=False, NtV_crit=-1.0):
        """ Condition model on the model side.

        Parameters
        ----------
        cov_pushfwd: tensor
            Pushforwarded covariance matrix, i.e. K F^T
        F
            Forward operator
        sigma0: float
            Standard deviation parameter for the kernel.
        m0: float
            Prior mean parameter.
        concentrate: bool
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood.
        NtV_crit: (deprecated)
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
        for attempt in range(10):
            try:
                u = torch.cholesky(inv_inversion_operator)
            except RuntimeError:
                print("Cholesky failed: Singular Matrix.")
                # Increase NtV in steps of 10%.
                newNtV = 1.1 * NtV
                print("Increasing NtV from original {} to {} and retrying.".format(NtV, newNtV))
                inv_inversion_operator = torch.add(
                    0.1 * NtV * self.data_ones,
                    inv_inversion_operator)
            else:
                break

        z2, _ = torch.triangular_solve(self.prior_misfit, u, upper=False)
        y2, _ = torch.triangular_solve(z2, u.t(), upper=True)
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

    def optimize(self, K_d, n_epochs, device, sigma0_init=None,
            lr=0.007, NtV_crit=-1.0):
        """ Given lambda0, optimize the two remaining hyperparams via MLE.
        Here, instead of giving lambda0, we give a (stripped) covariance
        matrix. Stripped means without sigma0.

        The user can choose between CPU and GPU.

        Parameters
        ----------
        K_d: tensor
            (stripped) Covariance matrix in data space.
        n_epochs: int
            Number of training epochs.
        device: Torch.device
            Device to use for optimization, either CPU or GPU.
        sigma0_init: float
            Starting value for gradient descent. If None, then use the value
            sotred by the model class (that is, the one resulting from the
            previous optimization run).
        lr: float
            Learning rate.
        NtV_crit: (deprecated)

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
                    concentrate=True)
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
                self.logger.info("sigma0: {}".format(self.sigma0.item()))
                self.logger.info("Log-likelihood: {}".format(log_likelihood.item()))
                self.logger.info("RMSE train error: {}".format(train_RMSE.item()))

        self.logger.info("Log-likelihood: {}".format(log_likelihood.item()))
        self.logger.info("RMSE train error: {}".format(train_RMSE.item()))

        # Send everything back to cpu.
        self.to_device(cpu)

        return

    def post_cov(self, cov_pushfwd, cells_coords, lambda0, sigma0, i, j, cl):
        """ Get posterior model covariance between two specified cells.

        Parameters
        ----------
        cov_pushfwd: tensor
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
        cl: module
            Covariance module implementing the kernel we are working with.
            The goal is to make this function kernel independent, though its a
            bit awkward and should be changed to an object oriented approach.

        Returns
        -------
        post_cov: float
            Posterior covariance between model cells i and j.

        """
        cov = cl.compute_cov(lambda0, cells_coords, i, j)

        post_cov = sigma0**2 * (cov -
                torch.mm(
                    sigma0**2 * cov_pushfwd[i, :].reshape(1, -1),
                    torch.mm(
                        self.inversion_operator, cov_pushfwd[j, :].reshape(-1, 1))))

        return post_cov

    def compute_post_cov_diag(self, cov_pushfwd, cells_coords, lambda0, sigma0, cl):
        """ Compute diagonal of the posterior covariance matrix.

        Parameters
        ----------
        cov_pushfwd: tensor
            Pushforwarded covariance matrix, i.e. K F^T
        cells_coords: tensor
            n_cells * n_dims: cells coordinates
        lambda0: float
            Lenght-scale parameter
        sigma0: float
            Prior variance.
        cl: module
            Covariance module implementing the kernel we are working with.
            The goal is to make this function kernel independent, though its a
            bit awkward and should be changed to an object oriented approach.

        Returns
        -------
        post_cov_diag: tensor
            Vector containing the posterior variance of model cell i as its
            i-th element.

        """
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

        Returns
        -------
        float
            RMSE on training set.

        """
        return torch.sqrt(torch.mean(
            (self.d_obs - self.mu_post_d)**2))

    def condition_number(self, cov_pushfwd, F, sigma0):
        """ Compute condition number of the inversion matrix R.

        Used to detect numerical instability.

        Parameters
        ----------
        cov_pushfwd: tensor
            Pushforwarded covariance matrix, i.e. K F^T
        F: tensor
            Forward operator
        sigma0: float
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

    def get_inversion_op_cholesky(self, K_d, sigma0):
        """ Compute the Cholesky decomposition of the inverse of the inversion operator.
        Increases noise level if necessary to make matrix invertible.

        Note that this method updates the noise level if necessary.

        Parameters
        ----------
        K_d: Tensor
            The (pushforwarded from model) data covariance matrix (stripped
            from sigma0).
        sigma0: Tensor
            The (model side) standard deviation.

        Returns
        -------
        Tensor
            L such that R = LL^t. L i lower triangular.

        """
        self.R = (self.data_std**2) * self.data_ones + sigma0**2 * K_d

        # Try to Cholesky.
        for attempt in range(50):
            try:
                L = torch.cholesky(self.R)
            except RuntimeError:
                self.logger.info("Cholesky failed: Singular Matrix.")
                # Increase noise in steps of 5%.
                self.data_std += 0.05 * self.data_std
                self.R = (self.data_std**2) * self.data_ones + sigma0**2 * K_d
                self.logger.info("Increasing data std from original {} to {} and retrying.".format(
                        self.data_std_orig, self.data_std))
            else:
                return L
        # If didnt manage to invert.
        raise ValueError(
            "Impossible to invert matrix, even at noise std {}".format(self.data_std))
        return -1
    
    def inv_op_vector_mult(self, x):
        """ Multiply a vector by the inversion operator, using Cholesky
        approach (numerically more stable than computing inverse).

        Parameters
        ----------
        x: Tensor
            The vector to multiply.

        Returns
        -------
        Tensor
            Multiplied vector R^(-1) * x.

        """
        z, _ = torch.triangular_solve(x, self.inv_op_L, upper=False)
        y, _ = torch.triangular_solve(z, self.inv_op_L.t(), upper=True)
        return y
