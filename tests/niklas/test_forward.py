""" Test the computation of the forward operator. In fact, check that we get
the same values as Niklas.

"""
# This script computes the Forward corresponding to Niklas's data.
import build_forward as bu

from nose.tools import assert_almost_equal

import h5py
import numpy as np


data_path = "/home/cedric/PHD/Dev/Volcano/tests/niklas/F_Niklas_raw.mat"


class TestForward():
    def setUp(self):
        dataset = h5py.File(data_path, 'r')

        N_OBS = 543
        N_MODEL = 179171
        self.F_niklas_raw = np.reshape(
                dataset['F/data'], (N_OBS, N_MODEL), order = 'F')

    def test_forward(self):
        """ Check forward agrees with Niklas.
        """
        # Here, compare with outputs from Niklas's code.
        F = bu.F

        a = F[47352, 40]
        assert_almost_equal(a, 0.001149142185113)

        a = F[117412, 500]
        assert_almost_equal(a, 0.01927512613701765)

        a = F[99449, 10]
        assert_almost_equal(a, 0.023827290251233535)
