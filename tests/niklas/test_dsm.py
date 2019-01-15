""" Test the niklas.dsm module.

"""
from volcapy.niklas import dsm as dsm_mod

import numpy as np

from numpy.testing import assert_array_equal

data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

def test_from_matfile():
    """ Build dsm from Niklas matfile.
    """
    dsm = dsm_mod.DSM.from_matfile(data_path)
