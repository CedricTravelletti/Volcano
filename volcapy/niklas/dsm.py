# File: dsm.py, Author: Cedric Travelletti, Date: 15.01.2019.
""" Class implementing dsm functionalities.
Also allows to build a dsm object from the raw Niklas data.

A dsm is basically a two dimensional array of cell, where for each cell we get
the midpoint along the x-y axis and the elevation.

Since we only have midpoints, and since the cells might have different sizes,
we also need a list of resolutions.

"""
import h5py
import numpy as np

# Import the definition of a Cell from inversion_grid, so that we have a
# coherent data format across modules.
from volcapy.niklas.inversion_grid import Cell


class DSM:
    """ DSM functionalities
    """
    def __init__(self, longs, lats, elevations, res_x, res_y):
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
        res_x: [float]
            For each x-cell, gives its size in meters.
        res_y: [float]
        """
        self.xs = longs
        self.ys = lats
        self.elevations = elevations

        self.dimx = len(self.xs)
        self.dimy = len(self.ys)

        self.res_x = res_x
        self.res_y = res_y

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

        # TODO: Clean this.
        # We have to specify the size of each dsm cell.
        # This could be computed automatically.
        # For the moment being, we hardcode the size of Niklas dsm here.
        dem_res_x = 50*[100] + 5*194*[10] + 70*[100]
        dem_res_y = 50*[100] + 5*190*[10] + 75*[100]

        return cls(xs, ys, elevations, dem_res_x, dem_res_y)

    def __getitem__(self, index):
        """ Return the coordinates/elecation of the cell at the given index.
        Also allows for slicing (i.e. giving an array of indices instead of a
        single scalar tuple.

        This returns a Cell object, to make the data format compatible with the
        inversion_grid module.
        """
        # If we get a tuple, then we simply have to return the single cell it
        # indexes.
        if isinstance(index, tuple):
            return self._get_individual_item(index[0], index[1])

        # If not, then we need to iterate the list we were provided, build once
        # cell each time, store them in a list and return the list.
        else:
            cells = []
            for ind in index:
                cells.append(self._get_individual_item(ind[0], ind[1]))

            return cells

    def _get_individual_item(self, i, j):
        """ Helper function for the above. Builds and return the cell
        corresponding the a single index. Then, we chain it with the above in
        order to allow the user to provide a list of indices and get back a
        list of cells.

        Parameters
        ----------
        i: int
            Index along the x-dimension.
        j: int
            Index along the y-dimension.

        Returns
        -------
        Cell
        """
        # Get the resolutions and lat/longs/elevations.
        res_x = self.res_x[i]
        res_y = self.res_y[j]
        x = self.xs[i]
        y = self.ys[j]
        elevation = self.elevations[i, j]

        # Create a cell and return it.
        # Note the difference between dsm an the inversion grid.
        # In the dsm we only get the midpoints, so we use the resolutions to
        # compute the boundaries of the cell.
        # Also, we only have one elevation, so we put zh to 0.
        return Cell(x - res_x/2.0, x + res_x/2.0,
                y - res_y/2.0, y+res_y/2.0, elevation, 0.0)
