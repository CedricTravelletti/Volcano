""" Make covariance matrix sequentially updatable.

The goal is that, instead of performing a conditioning on lots of measurement
in one go, we can instead chunk the data and perform several conditioning in
series, updating the covariance matrix along the way.

The CRUX of this module is that it handles covariance matrices that are too
large to fit in memory.

This is defined in the document "05-12-2019: cov_update_implementation".

Remark: Someday, we might want to switch to KeOps for the covariance matrix.

Concept
-------
Let F_1,..., F_n be measurement operators/forwards.
We want to compute the conditional covariance corresponding to those
measurements. We could stack the matrices an just condition on one big
measurement/forward. This would require us to inverse a matrix the size of the
whole dataset.

Alsok we work in the *big model* framework, i.e. when the model discretization
is too fine to allow covariance matrices to ever sit in memory (contrast this
with the *big data* settting.

TODO
----

Implement some shape getter.
Implement computation of the diagonal.

"""
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


class UpdatableCovariance:
    """ Covariance matrix that can be sequentially updated to include new
    measurements (conditioning).

    Attributes
    ----------
    pushforwards: List[Tensor]
        The covariance pushforwards corresponding to each conditioning step.
    inversion_ops: List[Tensor]
        The inversion operators corresponding to each conditioning step.

    """
    def __init__(self, cov_module, lambda0, sigma0, epsilon0, cells_coords):
        """ Build an updatable covariance from a traditional covariance module.

        Params
        ------

        """
        self.cov_module = cov_module
        self.lambda0 = lambda0
        self.sigma0 = sigma0
        self.epsilon0 = epsilon0
        self.cells_coords = cells_coords

        self.pushforwards = []
        self.inversion_ops = []

    def mul_right(self, A):
        """ Multiply covariance matrix from the right.
        Note that the matrix with which we multiply should map to a smaller
        dimensional space.

        Parameters
        ----------
        A: Tensor
            Matrix with which to multiply.

        Returns
        -------
        Tensor
            C * A

        """
        # First compute the level 0 pushforward.
        # IMPORTANT: FIX THE TRANSPOSING STUFF.
        cov_pushfwd_0 = self.cov_module.compute_cov_pushforward(
                self.lambda0, A.t(), self.cells_coords, gpu, n_chunks=200,
                n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            cov_pushfwd_0 += self.sigma0**2 * p @ (r @ (p.t() @ A))

        # Note the first term (the one with C_0 alone) only has one sigma0**2
        # factor associated with it, whereas all other terms in the updating
        # have two one.
        return self.sigma0**2 * cov_pushfwd_0

    def sandwich(self, A):
        """ Sandwich the covariance matrix on both sides.
        Note that the matrix with which we sandwich should map to a smaller
        dimensional space.

        Parameters
        ----------
        A: Tensor
            Matrix with which to sandwich.

        Returns
        -------
        Tensor
            A^t * C * A

        """
        # First compute the level 0 pushforward.
        cov_pushfwd_0 = A.t() @ self.cov_module.compute_cov_pushforward(
                self.lambda0, A, self.cells_coords, gpu, n_chunks=200,
                n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            tmp = p.t() @ A
            cov_pushfwd_0 += self.sigma0**2 * tmp.t() @ (r @ tmp)

        return self.sigma0**2 * cov_pushfwd_0

    def update(self, F):
        """ Update the covariance matrix / perform a conditioning.

        Params
        ------
        F: Tensor
            Measurement operator.

        """
        self.pushforwards.append(self.mul_right(F.t()))

        # Get inversion op by Cholesky.
        R = F @ self.pushforwards[-1]
        R = self.sigma0**2 * R + self.epsilon0**2 * torch.eye(F.shape[0])
        try:
            L = torch.cholesky(R)
        except RuntimeError:
            print("Error inverting.")
            print(F)

        inversion_op = torch.cholesky_inverse(L)
        self.inversion_ops.append(inversion_op)
