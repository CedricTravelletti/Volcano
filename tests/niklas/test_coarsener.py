""" Test the niklas.coarsener module.

"""
from volcapy.niklas import dsm as dsm_mod
from volcapy.niklas import coarsener as coas

import numpy as np

from numpy.testing import assert_array_equal


class TestCoarsener():
    def setUp(self):
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
        self.coarsener = coas.Coarsener(coarsen_x, coarsen_y, dsm)

    def tearDown(self):
        pass

    def test_indices_correspondence(self):
        """ Coarse cells to dsm index correspondence.
        """

        # Check the index correspondence.
        assert_array_equal(self.coarsener.inds_x, [[0], [1,2,3], [4, 5]])
        assert_array_equal(self.coarsener.inds_y, [[0, 1], [2, 3]])

    def test_get_fine_indices(self):
        """ Get fine indices corresponding to coarse cell.
        """
        inds = self.coarsener.get_fine_indices(1, 1)
        assert_array_equal(inds, [(1,2), (1,3), (2,2), (2,3), (3,2), (3,3)])

    def test_get_fine_elevations(self):
        """ Get fine elevations corresponding to coarse cell.
        """
        elevations = self.coarsener.get_fine_elevations(1, 1)
        assert_array_equal(elevations, [7,8,12,13,16,17])
