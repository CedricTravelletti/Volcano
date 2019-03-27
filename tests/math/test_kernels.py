""" Test the implementation of the Banerjee formula.

"""
import volcapy.math.kernels as krn

import numpy as np

from nose.tools import assert_almost_equal


def test_square_exp():
    """ Squared exponential kernel.
    """
    x = np.array([1.0,2.0,3.0])
    y = np.array([-2.0,4.0,5.0])

    a = krn.square_exp(x, y, 4.0, 2.0)

    assert_almost_equal(a, 0.057056937366724014)
