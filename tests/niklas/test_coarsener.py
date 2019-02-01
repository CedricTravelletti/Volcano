""" Test the niklas.coarsener module.

"""
from volcapy.niklas import dsm as dsm_mod
from volcapy.niklas import coarsener as coas

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal


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

        dem_res_x = 6*[1]
        dem_res_y = 4*[1]

        dsm = dsm_mod.DSM(longs, lats, elevations, dem_res_x, dem_res_y)

        coarsen_x = [1, 3, 2]
        coarsen_y = [2,2]

        res_x = [100, 30, 20]
        res_y = [20, 20]

        self.coarsener = coas.Coarsener(coarsen_x, coarsen_y, res_x, res_y, dsm)

    def tearDown(self):
        pass

    def test_indices_correspondence(self):
        """ Coarse cells to dsm index correspondence.
        """

        # Check the index correspondence.
        assert_array_equal(self.coarsener.fine_inds_x, [[0], [1,2,3], [4, 5]])
        assert_array_equal(self.coarsener.fine_inds_y, [[0, 1], [2, 3]])

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

    def test_get_coords(self):
        """ Get coordinates of midpoint in finer grid.
        """
        coords1 = self.coarsener.get_coords(0, 0)
        assert_equal(coords1, (1, 4.5))

        coords2 = self.coarsener.get_coords(1, 1)
        assert_equal(coords2, (3, 6.5))

    def test_get_fine_cells(self):
        """ Get fine cells corresponding to coarse cell.
        """
        cells = self.coarsener.get_fine_cells(1, 1)

        cell = cells[0]
        assert_equal((cell.x, cell.y, cell.z), (2, 6, 7))

        cell = cells[1]
        assert_equal((cell.x, cell.y, cell.z), (2, 7, 8))

        cell = cells[2]
        assert_equal((cell.x, cell.y, cell.z), (3, 6, 12))
        
        cell = cells[3]
        assert_equal((cell.x, cell.y, cell.z), (3, 7, 13))

        cell = cells[4]
        assert_equal((cell.x, cell.y, cell.z), (4, 6, 16))

        cell = cells[5]
        assert_equal((cell.x, cell.y, cell.z), (4, 7, 17))
