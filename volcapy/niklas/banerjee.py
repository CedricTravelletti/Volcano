# File: banerjee.py, Author: Cedric Travelletti, Date: 16.01.2019.
""" Implements the Banerjee formula for the gravitational field froduced by a
parallelepiped of uniform density.
"""
import numpy as np

def banerjee(xh, xl, yh, yl, zh, zl, x_data, y_data, z_data):
    """ Returns the gravity field (measured at a data point) produced by a
    parallelepiped of uniform unit density.

    Parameters
    ----------
    xh: float
        Location (in meters) of the upper (maximal x) corner of the
        parallelepiped.
    xl: float
        Same, but lower corner.
    yh: float
        Same but along y dimension.
    yl: float
        Same but along y dimension.
    zh: float
        Same but along z dimension.
    zl: float
        Same but along z dimension.
    x_data: float
        x coordinate (in meters) of the location at which we measure the field.
    y_data: float
    z_data: float

    """
    # Generate the different combinations we need.
    deltas_x = [xh - x_data, xl - x_data]
    deltas_y = [yh - y_data, yl - y_data]
    deltas_z = [zh - z_data, zl - z_data]

    B = 0
    for i, dx in enumerate(deltas_x):
        for j, dy in enumerate(deltas_y):
            for l, dz in enumerate(deltas_z):
                sign = (-1)**(i + j + l)
                B += _banerjee(dx, dy, dz, sign)
    return B

def _banerjee(x, y, z, sign):
    """ Helper function for readability.
    """
    R = np.sqrt(x**2 + y**2 + z**2)
    return(sign*x*np.log((R + y) / (R - y)) + sign*y*np.log((R + x) / (R - x))
            - 2*sign*z*np.arctan(x * y / (R * z)))
