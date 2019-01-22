# File: coarsener.py, Author: Cedric Travelletti, Date: 15.01.2019.
""" Class implementing coarsening functions.

THIS IS STRICTLY 2-DIMENSIONAL, i.e., this only refers to x-y slice.

We might want to have an inversion grid that is coarser than the dsm,
i.e., each cell in the inversion grid corresponds to several cells in the dsm
grid.
Then, for the surface cells, we have to determine how many z-levels are below
it. We look at the number of z-levels under each sub-cell making up the surface
cell and take the minimum one.

"""
import numpy as np


class Coarsener():
    """ Build a grid coarser than the dsm.

    Parameters
    ----------
    coarsen_x: [int]
        Defines the coarsening along the x-dimension.
        For example, if coarsen_x = [10, 5, 5, ...], then the first cells
        (along dim-x) of the coarser grid will correspond to the first 10 cells
        of the dsm, then, the second cells (along dim-x) will correspond to the
        next 5 cells, and so on.
    coarsen_y: [int]
    res_x: List[float]
        Size of each cell in meters.
    res_y: List[float]
    dsm: DSM
    """
    def __init__(self, coarsen_x, coarsen_y, res_x, res_y, dsm):
        self.dimx = len(coarsen_x)
        self.dimy = len(coarsen_y)

        self.coarsen_x = coarsen_x
        self.coarsen_y = coarsen_y

        self.res_x = res_x
        self.res_y = res_y

        self.dsm = dsm

        # Check dimensions.
        if not (sum(coarsen_x) == dsm.dimx and sum(coarsen_y) == dsm.dimy):
            raise ValueError("Coarsening doesnt agree with dimensions of dsm.")

        if not (len(res_x) == len(coarsen_x) and len(res_y) == len(coarsen_y)):
            raise ValueError(
            "Length of resolution vectors differs from length of coarsening vector")


        # Produce index correspondances.
        # This will be a list of lists. Each element contains a list of the
        # indices in the big grid that correspond to that element.
        self.inds_x = []
        self.inds_y = []

        # Count how many cells in the big table we have already passed.
        count = 0

        # Loop over each cell in the coarse gird.
        # (Each has its own coarsening degree).
        for coarsening in coarsen_x:
            # Indices corresponding to current coarse cell.
            tmp = []
            for k in range(coarsening):
                tmp.append(count + k)

            self.inds_x += [tmp]
            count += coarsening

        count = 0
        for coarsening in coarsen_y:
            # Indices corresponding to current coarse cell.
            tmp = []
            for k in range(coarsening):
                tmp.append(count + k)

            self.inds_y += [tmp]
            count += coarsening

    def get_fine_indices(self, i, j):
        """ Get the indices (in the finer grid) of cells correspondin to cell
        (i, j) in the coarser grid.

        Parameters
        ----------
        i,j: int
            Index in the coarse grid.

        Returns
        -------
        List[(int, int)]
            List of indices in the bigger grid.

        """
        # Get the x and y index lists correspondin to the cell.
        inds_x = self.inds_x[i]
        inds_y = self.inds_y[j]

        # Return a list of indices in the finer grid.
        fine_indices = []
        for x in inds_x:
            for y in inds_y:
                fine_indices.append((x, y))
        return fine_indices

    def get_fine_elevations(self, i, j):
        """ Get the elevations (in the finer grid) of cells corresponding to cell
        (i, j) in the coarser grid.

        Parameters
        ----------
        i,j: int
            Index in the coars grid.

        Returns
        -------
        List[float]
            List of elevations in the bigger grid.

        """
        fine_indices = self.get_fine_indices(i, j)
        elevations = []
        for cell in fine_indices:
            elevation = self.dsm[cell[0], cell[1]][2]
            elevations.append(elevation)
        return elevations

    def get_coords(self, i, j):
        """ Get lat/long of the current cell in the coarse grid.
        We use the mean of the coordinates of the cell in the larger grid that
        correspond to the cell under consideration.
        """
        # Get the indexes of the corresponding cells in the big grid.
        fine_indices = self.get_fine_indices(i, j)

        # For each of these cells, get their x and y coordinates.
        # Put in a list.
        coord_x = []
        coord_y = []
        for ind in fine_indices:
            coords = self.dsm[ind[0], ind[1]]

            coord_x.append(coords[0])
            coord_y.append(coords[1])

        return(np.mean(coord_x), np.mean(coord_y))

    def get_fine_cells(self, i, j):
        """ Get the cell (in the finer grid) corresponding to cell
        (i, j) in the coarser grid.

        Parameters
        ----------
        i,j: int
            Index in the coarse grid.

        Returns
        -------
        List[float]
            List of elevations in the bigger grid.

        """
        fine_indices = self.get_fine_indices(i, j)
        cells = []
        for ind in fine_indices:
            cell = self.dsm[ind[0], ind[1]]
            cells.append(cell)
        return cells
