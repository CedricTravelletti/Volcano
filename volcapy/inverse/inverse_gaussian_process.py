"""
HEAVY REFACTOR OF THE GP CLASS.

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

Important Implementation Detail
-------------------------------
Products of vectors with the inversion operator R^{-1} * x should be computed
using inv_op_vector_mult(x).

Conditioning
------------
Conditioning always compute the inversion operator, and m0 via concentration.
Hence, every call to conditioning (condition_data or condition_model), will
update the following attributes:
    - inv_op_L
    - inversion_operator
    - m0
    - mu0_d
    - prior_misfit
    - weights
    - mu_post_d (TODO: maybe shouldn't be an attribute of the GP.)

"""
import numpy as np
import torch


class InverseGaussianProcess(torch.nn.Module):
    """

    Attributes
    ----------
    m0: float
        Current value of the prior mean of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    sigma0: float
        Current value of the prior standard deviation of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    lambda0: float
        Current value of the prior lengthscale of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    cov_module: CovarianceModule
        Covariance kernel to use
    output_device: toch.device
        GPU to use to store the main results. Defaults to GPU0.
    logger
        An instance of logging.Logger, used to output training progression.

    """
    def __init__(self, m0, sigma0, lambda0,
            cells_coords, cov_module,
            output_device=None,
            logger=None):
        """

        Parameters
        ----------
        m0: float
            Prior mean.
        sigam0: float
            Prior standard deviation.
        lambda0: float
            Prior lengthscale.
        cells_coords: tensor
            Models cells, fixed once and for all.
        cov_module: CovarianceModule
            Kernel to use.
        logger: Logger

        """
        super(InverseGaussianProcess, self).__init__()
        
        # Get GPUS.
        if output_device is None:
            try:
                output_device = torch.device('cuda:0')
            except:
                raise ValueError("No GPU detected. Volcapy needs a GPU to run. Aborting.")
        self.output_device = output_device

        self.m0 = m0
        self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0)).to(output_device)
        self.lambda0 = lambda0

        self.cells_coords = cells_coords.to(output_device)
        self.n_model = cells_coords.shape[0]

        self.kernel = cov_module

        # The heavy part is to compute the pushforward. It is needed by all
        # other operations. Hence, if it has already been computed by an
        # operation, just leave it as is.
        self.is_precomputed_pushfw = False

        # If no logger, create one.
        if logger is None:
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        self.logger = logger

    def compute_pushfwd(self, G):
        """ Given a measurement operator, compute the associated covariance
        pushforward K G^T.

        """
        # Compute the compute_covariance_pushforward and data-side covariance matrix
        self.pushfwd = self.kernel.compute_cov_pushforward(
                self.lambda0, G, self.cells_coords,
                self.output_device, n_chunks=200, n_flush=50)
        self.K_d = G @ self.pushfwd

    def condition_data(self, G, y, data_std, concentrate=False,
            is_precomp_pushfwd=False):
        """ Given a bunch of measurement, condition model on the data side.
        I.e. only compute the conditional law of the data vector G Z, not of Z
        itself.

        Parameters
        ----------
        G: tensor
            Measurement matrix
        y: tensor
            Observed data
        data_std: flot
            Data noise standard deviation.
        concentrate: bool
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood instead of using the current value of the
            hyperparameter.
        is_precomp_pushfwd: bool
            Set to True if the covariance pushforward has already been computed
            by a previous operation.

        Returns
        -------
        mu_post_d: tensor
            Posterior mean data vector

        """
        if not is_precomp_pushfwd:
            self.compute_pushfwd(G)

        # Get Cholesky factor (lower triangular) of the inversion operator.
        self.inv_op_L = self.get_inversion_op_cholesky(self.K_d, self.sigma0)
        self.inversion_operator = torch.cholesky_inverse(self.inv_op_L)
            
        if concentrate:
            # Determine m0 (on the model side) from sigma0 by concentration of the Ll.
            m0 = self.concentrate_m0()
        else: m0 = self.m0

        # Prior mean (vector) on the data side.
        mu0_d_stripped = torch.mm(G, torch.ones((self.n_model, 1),
                dtype=torch.float32))
        mu0_d = m0 * mu0_d_stripped
        prior_misfit = y - mu0_d

        self.weights = self.inv_op_vector_mult(prior_misfit)

        m_post_d = mu0_d + torch.mm(self.sigma0**2 * self.K_d, self.weights)

        return m_post_d

    def neg_log_likelihood(self, y):
        """ Computes the negative log-likelihood of the current state of the
        model.
        Note that this function should be called AFTER having run a
        conditioning, since it depends on the inversion operator computed
        there.

        Params
        ------
        y: tensor
            Data vector
        Returns
        -------
        float

        """
        # WARNING!!! determinant is not linear! Taking constants outside adds
        # power to them.
        log_det = torch.logdet(self.R)
        prior_misfit = y - mu0_d
        weights = self.inv_op_vector_mult(prior_misfit)
        nll = log_det + torch.mm(prior_misfit.t(), weights)

        return nll

    def concentrate_m0(self, y):
        """ Compute m0 (prior mean parameter) by MLE via concentration.

        Note that the inversion operator should have been updated first.

        """
        # Prior mean (vector) on the data side.
        mu0_d_stripped = torch.mm(G, torch.ones((self.n_model, 1),
                dtype=torch.float32))
        # Compute R^(-1) * G * I_m.
        tmp = self.inv_op_vector_mult(mu0_d_stripped)
        conc_m0 = (y.t() @ tmp) / (mu0_d_stripped.t() @ tmp)

        return conc_m0

    def condition_model(self, G, y, data_std, concentrate=False,
            is_precomp_pushfwd=False):
        """ Given a bunch of measurement, condition model on the data side.
        I.e. only compute the conditional law of the data vector G Z, not of Z
        itself.

        Parameters
        ----------
        G: tensor
            Measurement matrix
        y: tensor
            Observed data
        data_std: flot
            Data noise standard deviation.
        concentrate: bool
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood instead of using the current value of the
            hyperparameter.
        is_precomp_pushfwd: bool
            Set to True if the covariance pushforward has already been computed
            by a previous operation.

        Returns
        -------
        mu_post_m
            Posterior mean model vector
        mu_post_d
            Posterior mean data vector

        """
        # Conditioning model is just conditioning on data and then computing
        # posterior mean and (co-)variance on model side.
        m_post_d = condition_data(G, y, data_std, concentrate=concentrate,
                is_precomp_pushfwd=is_precomp_pushfwd)

        # Posterior model mean.
        # Can re-use the m0 and weights computed by condition_data.
        if concentrate:
            m0 = self.concentrate_m0()
        else: m0 = self.m0

        m_post_m = (
                m0 * torch.ones((self.n_model, 1))
                + (self.sigma0**2 * self.pushfwd @ self.weights))

        return m_post_m.detach(), m_post_d

    def train_fixed_lambda(lambda0, G, y, data_std,
            n_epochs, lr=0.007):
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
        # Compute the pushforward once and for all, since it only depends on
        # lambda0 and G.
        self.lambda0 = lambda0
        self.compute_pushfwd(G)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            # Forward pass: Compute predicted y by passing
            # x to the model
            m_post_d = condition_data(G, y, data_std, concentrate=True,
                is_precomp_pushfwd=True)
            log_likelihood = self.neg_log_likelihood()

            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            log_likelihood.backward(retain_graph=True)
            optimizer.step()

            # Periodically print informations.
            if epoch % 100 == 0:
                # Compute train error.
                train_RMSE = torch.sqrt(torch.mean(
                        (y- m_post_d)**2))
                self.logger.info("sigma0: {}".format(self.sigma0.item()))
                self.logger.info("Log-likelihood: {}".format(log_likelihood.item()))
                self.logger.info("RMSE train error: {}".format(train_RMSE.item()))

        self.logger.info("Log-likelihood: {}".format(log_likelihood.item()))
        self.logger.info("RMSE train error: {}".format(train_RMSE.item()))
        self.logger.info(self.parameters())

        return

    def get_inversion_op_cholesky(self, K_d, data_std):
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
            L such that R = LL^t. L is lower triangular.

        """
        data_std_orig = data_std
        n_data = K_d.shape[0]
        data_ones = torch.eye(n_data, dtype=torch.float32)
        self.R = (data_std**2) * data_ones + self.sigma0**2 * K_d

        # Check condition number if debug mode on.
        if __debug__:
            self.logger.info(
                    "Condition number of (inverse) inversion operator: {}".format(
                    np.linalg.cond(self.R.detach().numpy())))

        # Try to Cholesky.
        for attempt in range(50):
            try:
                L = torch.cholesky(self.R)
            except RuntimeError:
                self.logger.info("Cholesky failed: Singular Matrix.")
                # Increase noise in steps of 5%.
                data_std += 0.05 * data_std
                self.R = (data_std**2) * data_ones + self.sigma0**2 * K_d
                self.logger.info("Increasing data std from original {} to {} and retrying.".format(
                        data_std_orig, data_std))
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
