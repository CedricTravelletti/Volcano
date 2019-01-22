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

        dem_res_x = 6*[1]
        dem_res_y = 4*[1]

        dsm = dsm_mod.DSM(longs, lats, elevations, dem_res_x, dem_res_y)

        coarsen_x = [1, 3, 2]
        coarsen_y = [2,2]

        res_x = [100, 30, 20]
        res_y = [20, 20]

        z_levels = [-1, 0, 2, 3, 4, 5]
        self.inversion_grid = InversionGrid(coarsen_x, coarsen_y, res_x, res_y,
                z_levels, dsm)

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

    def test_resolutions(self):
        """ Check resolution of coarse cells correctly defined.
        """
        for cell in self.inversion_grid:
            # Look for a specific cell whose res we know.
            if cell.x == 3.0 and cell.y == 4.5:
                assert_equal(cell.res_x, 30)
                assert_equal(cell.res_y, 20)

            if cell.x == 5.5 and cell.y == 6.5:
                assert_equal(cell.res_x, 20)
                assert_equal(cell.res_y, 20)

    def test_z_resolutions(self):
        """ Check vertical resolutions are computed correctly.
        """
        z_res = [1, 2, 1, 1, 1, 0]
        assert_equal(self.inversion_grid.z_resolutions, z_res)

    def test_topmost_ind_to_2d_ind(self):
        """ Check conversion of 1D index to 2D one for topmost cells.
        """
        ind = self.inversion_grid.topmost_ind_to_2d_ind(3)
        assert_equal(ind, (1, 1))

        ind = self.inversion_grid.topmost_ind_to_2d_ind(4)
        assert_equal(ind, (2, 0))

    def test_fine_cells_from_topmost_ind(self):
        """ Check the fine cells corresponding to a given index of a topmost cell.
        """
        cells = self.inversion_grid.fine_cells_from_topmost_ind(2)
        assert_equal(cells[0].x, 2)
        assert_equal(cells[0].y, 4)
        assert_equal(cells[0].z, 5)

        assert_equal(cells[5].x, 4)
        assert_equal(cells[5].y, 5)
        assert_equal(cells[5].z, 15)
