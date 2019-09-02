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

    def __init__(self, mean, variance, covariance_func):
        self.mean = mean
        self.var = variance
        self.cov = covariance_func

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
        return (1 - norm.cdf(threshold, loc=self.mean[i],
                scale=np.sqrt(self.var[i])))

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

    def vorobev_quantile_inds(self, alpha, threshold):
        """ Returns Vorobev quantile alpha. In facts, return the indices of the
        points which are in the quantile.

        Parameters
        ----------
        alpha: float
            Level of the quantile to return.
            Will return points that have a prob greater than alpha to be in the
            excursion set.
        threshold: float
            Excursion threshold.

        Returns
        -------
        List[int]
            List of the indices of the points that are in the Vorobev quantile.

        """
        excursion_probs = self.compute_excursion_probs(threshold)

        # Indices of the points in the Vorob'ev quantile.
        # Warning: where return a tuple. Here we are 1D so get the first
        # element.
        vorobev_inds = np.where(excursion_probs > alpha)[0]

        return vorobev_inds

    def vorobev_expectation_inds(self, threshold):
        """ Same as above, but return Vorob'ev expectation this time.
        Parameters
        ----------
        threshold: float
            Excursion threshold.

        Returns
        -------
        List[int]
            List of the indices of the points that are in the Vorobev quantile.

        """
        excursion_probs = self.compute_excursion_probs(threshold)
        expected_measure = self.expected_excursion_measure(threshold)
        print("Expected measure: {}".format(expected_measure))

        # TODO: Refactor into dichotomic search.
        # Loop over confidence levels, increase till smaller than expected
        # measure.
        for alpha in np.linspace(0.0, 1.0, num=10):
            # Get the cell belonging to quantile alpha.
            vorb_inds = self.vorobev_quantile_inds(alpha, threshold)

            # Count how many, to get the measure.
            measure = np.count_nonzero(vorb_inds)

            # If smaller than expected measure, then continue, else return.
            if measure < expected_measure:
                print("alpha: {}".format(alpha))
                return vorb_inds

    # TODO: The following behaves as if all cells had size one. In the future:
    # should either subset cells when building the GP so that have all same
    # size, or loop over cells and get their resolutions.
    def expected_excursion_measure(self, threshold):
        """ Returns the expected measure of the excursion set above the
        threshold.

        """
        excursion_probs = self.compute_excursion_probs(threshold)
        return np.sum(excursion_probs)

    def vorobev_deviation(self, set_inds, threshold):
        """ Compute the Vorob'ev deviation of a given set.
        Parameters
        ----------
        set_inds: List[int]
            Indices of the cells belonging to the set.
        threshold: float
            Excursion threshold.

        Returns
        -------
        float
            Vorob'ev deviation of the set.

        """
        excursion_probs = self.compute_excursion_probs(threshold)
        a = 1 - excursion_probs[set_inds]
        b = np.delete(excursion_probs, set_inds)
        dev = np.sum(a) + np.sum(b)
        return dev
