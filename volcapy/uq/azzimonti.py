# File: azzimonti.py, Author: Cedric Travelletti, Date: 28.01.2019.
""" Module implementing estimation of excursion sets and uncertainty
quantification on them.

"""
import numpy as np
from scipy.stats import norm


class GaussianProcess():
    """ Implementation of Gaussian Process.

    The underlying spatial structure is just a list of points, that is, we do
    not need to know the real spatial structure, the GP only know the
    mean/variance/covariance at points number i or j.

    Parameters
    ----------
    mean: 1D array-like
        List (or ndarray). Element i gives the mean at point i.
    variance: 1D array-like
        Variance at every point.
    covariance_func: function
        Two parameter function. F(i, j) should return the covariance between
        points i and j.

    """

    def __init__(self, mean, variance, covariance):
        self.mean = mean
        self.var = variance
        self.cov = covariance

        self.dim = len(mean)

    def coverage_fct(self, i, threshold):
        """ Coverage function (excursion probability) at a point.

        Given a point in space, gives the probability that the value of the GP
        at that point is above some threshold.

        Parameters
        ----------
        i: int
            Index of the point to consider.
        threshold: float

        Returns
        -------
        float
            Probability that value of the field at point is above the
            threshold.
        """
        return (1 - norm.cdf(threshold, loc=self.mean[i], scale=self.var[i]))

    def compute_excursion_probs(self, threshold):
        """ Computes once and for all the probability of excursion above
        threshold for every point.

        Parameters
        ----------
        threshold: float

        Returns
        -------
        List[float]
            Excursion probabilities. Element i contains excursion probability
            (above threshold) for element i.

        """
        excursion_probs = np.zeros(self.dim)
        # Loop over all cells.
        for i in range(self.dim):
            excursion_probs[i] = self.coverage_fct(i, threshold)

        return excursion_probs
