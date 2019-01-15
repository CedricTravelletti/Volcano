""" Test the niklas.coarsener module.

"""
from volcapy.niklas import dsm as dsm_mod
from volcapy.niklas import coarsener as coas

import numpy as np

from numpy.testing import assert_array_equal

data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

def test_indices_correspondence():
    """ Coarse cells to dsm index correspondence.
    """
    longs = [1,2,3,4,5,6]
    lats = [4,5,6,7]
    elevations = np.array([
            [1,2,3,4],
            [5, 6, 7, 8 ],
            [10, 11, 12, 13],
            [14, 15, 16, 17],
            [18, 19, 20, 21],
            [22, 23, 24, 25],
            ])
    dsm = dsm_mod.DSM(longs, lats, elevations)

    coarsen_x = [1, 3, 2]
    coarsen_y = [2,2]
    coarsener = coas.Coarsener(coarsen_x, coarsen_y, dsm)

    # Check the index correspondence.
    assert_array_equal(coarsener.inds_x, [[0], [1,2,3], [4, 5]])
    assert_array_equal(coarsener.inds_y, [[0, 1], [2, 3]])
