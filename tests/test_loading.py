""" Test the loading module.

"""
from volcapy import loading as ld
from volcapy import grid as gd
from volcapy import matrix as mt

import numpy as np

from numpy.testing import assert_array_equal

data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"


def test_loading():
    data = ld.load_niklas(data_path)

def test_regularize_grid():
    unreg_grid = [
            [0, 0, 0],
            [1, 1, 1]]
    dx = 1.0
    dy = 1.0
    dz = 0.25

    spacings = (dx, dy, dz)

    # How the result should look.
    reg_grid = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0.25],
            [1, 0, 0.25],
            [0, 1, 0.25],
            [1, 1, 0.25],
            [0, 0, 0.5],
            [1, 0, 0.5],
            [0, 1, 0.5],
            [1, 1, 0.5],
            [0, 0, 0.75],
            [1, 0, 0.75],
            [0, 1, 0.75],
            [1, 1, 0.75],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]])

    result, dims = gd.regularize_grid(unreg_grid, spacings)

    assert_array_equal(result, reg_grid)
