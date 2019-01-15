# File: coarsener.py, Author: Cedric Travelletti, Date: 15.01.2019.
""" Class implementing coarsening functions.

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
    dsm: DSM
    """
    def __init__(self, coarsen_x, coarsen_y, dsm):
        self.dimx = len(coarsen_x)
        self.dimy = len(coarsen_y)
    
        # Check dimensions.
        if not (sum(coarsen_x) == dsm.dimx and sum(coarsen_y) == dsm.dimy):
            raise ValueError("Coarsening doesnt agree with dimensions of dsm.")
    
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
