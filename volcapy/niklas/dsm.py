# File: dsm.py, Author: Cedric Travelletti, Date: 15.01.2019.
""" Class implementing dsm functionalities.
Also allows to build a dsm object from the raw Niklas data.
"""
import h5py
import numpy as np


class DSM:
    """ DSM functionalities
    """
    def __init__(self, longs, lats, elevations):
        """ Default constructor to create dsm from a list of x-coordinates
        (longitudes), y-coordinates (latitudes) and a matrix of elevations
        (first coordinate for x-axis).

        Parameters
        ----------
        longs: [float]
        lats: [float]
        elevations [[float]]
            2D array, elevations[i, j] gives the elevation of the cell with
            coordinates longs[i], lats[j].
        """
        self.xs = longs
        self.ys = lats
        self.elevations = elevations

        self.dimx = len(self.xs)
        self.dimy = len(self.ys)

    @classmethod
    def from_matfile(cls, path):
        """ Construct from matlab data.

        Parameters
        ----------
        path: string
            Path to .mat file. Data inside should have the same format as
            provided by Niklas.

        """
        dataset = h5py.File(path, 'r')

        # DSM
        # We have arrays of arrays, so we flatten to be one dimensional.
        xs = np.ndarray.flatten(np.array(dataset['x']))
        ys = np.ndarray.flatten(np.array(dataset['y']))
        elevations = np.array(dataset['z'])

        # Build a dsm matrix.
        dsm = []
        for i in range(xs.size):
            for j in range(ys.size):
                dsm.append([xs[i], ys[j], elevations[i, j]])

        dsm = np.array(dsm)

        return cls(xs, ys, elevations)

    def __getitem__(self, index):
        """ Make class subscriptable.
        """
        return self.elevations[index]
