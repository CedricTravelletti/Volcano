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

"""


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
    def __init__(self, cov_module, lengthscale, coords):
        """ Build an updatable covariance from a traditional covariance module.

        Params
        ------

        """

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
        cov_pushfwd_0 = cl.compute_cov_pushforward(
                lambda0, A, cells_coords, gpu, n_chunks=200,
                n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            cov_pushfwd_0 += p @ (r @ (p.transpose() @ A))

        return res

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
        cov_pushfwd_0 = A.transpose() @ cl.compute_cov_pushforward(
                lambda0, A, cells_coords, gpu, n_chunks=200,
                n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            tmp = p.transpose() @ A
            res += tmp.transpose() @ (r @ tmp)

        return res

    def update(self, F):
        """ Update the covariance matrix / perform a conditioning.

        Params
        ------
        F: Tensor
            Measurement operator.

        """
        self.pushforwards.append(self.mul_right(F))

        # Get inversion op by Cholesky.
        R = F.transpose @ self.pushforwards[-1]
        L = torch.cholesky(R)
        inversion_op = torch.cholesky_inverse(L)
        self.inversion_ops.append(inversion_op)
