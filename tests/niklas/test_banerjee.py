""" Test the implementation of the Banerjee formula.

"""
from volcapy.niklas.banerjee import banerjee

import numpy as np

from nose.tools import assert_almost_equal


def test_from_matfile():
    """ Banerjee formula for whole parallelepiped.
    """
    # Here, compare with outputs from Niklas's code.
    a = banerjee(1,2,3,4,5,6,7,8,9)
    assert_almost_equal(a, -0.014082478)

    a = banerjee(-100,2,3,4,-2,6,7,100,9)
    assert_almost_equal(a, -0.00824017)
