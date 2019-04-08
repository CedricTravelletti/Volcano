""" Test the volcapy.grid.regridding module.

"""
from volcapy.grid.regridding import irregular_regrid_single_step, regrid_forward
from volcapy.inverse.flow import InverseProblem
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal


data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"


class TestCoarsener():
    def setUp(self):
        inverseProblem = InverseProblem.from_matfile(data_path)
        self.cells_coords = inverseProblem.cells_coords
        self.F = inverseProblem.forward

    def tearDown(self):
        pass

    def test_run(self):
        """ Check that the regridding runs.
        """
        coarse_cells_coords, coarse_to_fine_inds = irregular_regrid_single_step(
                self.cells_coords, 50.0)
        F_new = regrid_forward(self.F, coarse_to_fine_inds)

