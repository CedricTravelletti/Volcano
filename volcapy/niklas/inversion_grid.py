# File: inversion_grid.py, Author: Cedric Travelletti, Date: 15.01.2019.
""" Class implementing inversion grid.

The inversion grid has two importan properties

* It has a coarser resolution than the dsm, meaning that a single cell in the
 inversion grid corresponds to several cells in the dsm grid.
 * It is irregular, i.e. it doesn't span an entire parallelepiped.
   This is due to the fact that we don't include cells outside the volcano (in
   the air).

"""
import numpy as np
from volcapy.niklas.coarsener import Coarsener


class Cell():
    """ Class representing a cell in the inversion grid.
    """
    def __init__(self, x, y, z, res_x, res_y, res_z):
        self.x = x
        self.y =y
        self.z = z
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z

    # Redefine printing method so it display useful informations.
    def __str__(self):
        return("x: {} y: {} z: {} res_x: {} res_y: {} res_z: {}".format(self.x, self.y,
                self.z, self.res_x, self.res_y, self.res_z))

from collections.abc import Sequence
class InversionGrid(Sequence):
    def __init__(self, coarsen_x, coarsen_y, res_x, res_y, zlevels, dsm):
        """
        Parameters
        ----------
        coarsen_x: List[int]
            Defines the coarsening along the x-dimension.
            For example, if coarsen_x = [10, 5, 5, ...], then the first cells
            (along dim-x) of the coarser grid will correspond to the first 10 cells
            of the dsm, then, the second cells (along dim-x) will correspond to the
            next 5 cells, and so on.
        coarsen_y: List[int]
        res_x: List[float]
            Size of each cell in meters.
        res_y: List[float]
        zlevels: List[float]
            List of heights (in meters) at wich we place cells.
        dsm: DSM

        """
        self.coarsener = Coarsener(coarsen_x, coarsen_y, res_x, res_y, dsm)
        self.dimx = self.coarsener.dimx
        self.dimy = self.coarsener.dimy
        self.zlevels = zlevels

        # Create a vector giving the vertical size of each z_level.
        # This is used to define the z_res of each cells.
        # We put a resolution of zero to the topmost cells.
        self.z_resolutions = self.build_z_resolutions(zlevels)

        # For each cell in the horizontal plane, determine the maximal
        # altitude.
        self.grid_max_zlevel = self.build_max_zlevel(self.coarsener, self.zlevels)

        # Will be created when we call fill_grid.
        self.cells = []
        self.topmost_indices = []

        # Create the grid.
        self.fill_grid()

        # Call parent constructor.
        super().__init__()

    def __getitem__(self, i):
        return self.cells[i]

    def __len__(self):
        return len(self.cells)

    @staticmethod
    def build_max_zlevel(coarsener, zlevels):
        # Loop over coarse grid.
        # Here we just loop over the coarsening list (each element in there
        # corresponds to one cell in the coars grid) and use the index in the
        # list as an index in the coarse grid.
        grid_max_zlevel = np.zeros((coarsener.dimx, coarsener.dimy))

        for i, x in enumerate(coarsener.coarsen_x):
            for j,y in enumerate(coarsener.coarsen_y):
                # Get the smallest elevation.
                min_elevation = min(
                        coarsener.get_fine_elevations(i, j))

                # List of levels that are below the min elevation.
                levels_below = [v for v in zlevels if v < min_elevation]

                # If empty, then say we have at least one cell, at the minimal
                # level.
                if len(levels_below) == 0:
                    grid_max_zlevel[i, j] = min(zlevels)

                # Otherwise take the biggest one.
                else:
                    grid_max_zlevel[i, j] = max(levels_below)

        return grid_max_zlevel

    def fill_grid(self):
        """ Create the cells in the grid, taking into account the fact that the
        grid is irregulat, i.e., the number a z-floors can change, since we do
        not include cells that are 'in the air' wrt the dsm.
        """
        self.cells = []
        self.topmost_indices = []
        for i, res_x in enumerate(self.coarsener.res_x):
            for j,res_y in enumerate(self.coarsener.res_y):
                # Get the levels (number of floors) for that cell.
                current_max_zlevel = self.grid_max_zlevel[i, j]

                # CLUMSY: Fucking vertical resolution: we have to keep the
                # size information of each level, hence the intricate code
                # below.
                current_zlevels = [v for v in zip(self.zlevels,
                        self.z_resolutions) if v[0] <=
                        current_max_zlevel]

                # Loop over the floors, create the cells and append to list.
                # We want to retain the indices of the topmost cells,
                # below is a clever trick to do it.

                # CLUMSY: (see above remark) since we have a list of tuples, we
                # need to specify that we want to sort according to the first
                # entry.
                for z in sorted(current_zlevels, key=lambda x: x[0]):
                    x, y = self.coarsener.get_coords(i, j)
                    cell = Cell(x, y, z[0], res_x, res_y, z[1])
                    self.cells.append(cell)

                # Trick: since we sorted the list of z-levels, the last one to
                # get appended to the list is the one with the maximal
                # altitude, i.e. the topmost one. We can thus get its index by
                # looking at the length of the list.
                self.topmost_indices.append(len(self.cells) - 1)

        # We cast to numpy array, so that we can index also with lists.
        self.cells = np.array(self.cells)

    @staticmethod
    def build_z_resolutions(zlevels):
        """ Create a vector giving the vertical size of each z_level.
        This is used to define the z_res of each cells.
        We will put a resolution of zero to the topmost cells.

        """
        # Make a copy of the input so we do not modify it.
        zl = zlevels[:]
        tmp = zl + [zl[-1]]
        
        z_resolutions = zl
        for i, z in enumerate(zl):
            z_resolutions[i] = tmp[i + 1] - tmp[i]
        return z_resolutions
