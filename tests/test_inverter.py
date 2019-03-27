""" Test the Inverter/InverseProblem class
"""
from volcapy.flow import InverseProblem


def test_init_from_matfile():
    """ Building an InverseProblem from a matfile.
    """
    my_inverse_prob = InverseProblem.from_matfile(
            "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat")
