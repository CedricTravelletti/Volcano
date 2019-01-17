""" Test the niklas.inversion_grid module.

"""
from volcapy.niklas import dsm as dsm_mod
from volcapy.niklas.inversion_grid import InversionGrid

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal


class TestCoarsener():
    def setUp(self):
        longs = [1,2,3,4,5,6]
        lats = [4,5,6,7]
        elevations = np.array([
                [-2,2,3,4],
                [5, 6, 7, 8 ],
                [10, 11, 12, 13],
                [14, 15, 16, 17],
                [18, 19, 20, 21],
                [22, 23, 24, 25],
                ])
        dsm = dsm_mod.DSM(longs, lats, elevations)

        coarsen_x = [1, 3, 2]
        coarsen_y = [2,2]

        z_levels = [-1, 0, 2, 3, 4, 5]
        self.inversion_grid = InversionGrid(coarsen_x, coarsen_y, z_levels,
                dsm)

    def tearDown(self):
        pass

    def test_build_max_zlevels(self):
        """ Determination of maximal zlevels on inversion grid.
        """
        # What we should get.
        grid_max_zlevels = np.array([
                [-1, 2],
                [4, 5],
                [5, 5],
                ])

        # Check the index correspondence.
        assert_array_equal(self.inversion_grid.grid_max_zlevel,
                grid_max_zlevels)
