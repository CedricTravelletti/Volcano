# fILe: inversion_grid.py, Author: Cedric Travelletti, Date: 15.01.2019.
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
from volcapy.niklas.cell import Cell


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

        # It is important that the levels are in increasing order.
        self.zlevels = sorted(zlevels)

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

    def fill_grid(self):
        """ Create the cells in the grid, taking into account the fact that the
        grid is irregulat, i.e., the number a z-floors can change, since we do
        not include cells that are 'in the air' wrt the dsm.
        """
        topcells = []
        self.topmost_indices = []

        # --------------------------------------------------
        # BUILD TOPMOST CELLS
        # --------------------------------------------------
        # We do a first pass to put the topmost ones at the beginning of the
        # list.
        #
        # Note that these cell do not follow the vertical z-splitting of the
        # other one. That is, they have their true altitude as altitude.
        for i in range(self.dimx):
            for j in range(self.dimy):

                # TODO: Maybe needs to be refactored.
                # Add a new attribute to the topmost inversion cells:
                # Each one stores a list of the fine cells that make it up.
                # This takes some memory, but will speed up the refinement
                # process: all information will be directly available, no
                # lookup necessary.
                cell = self.coarsener.get_coarse_cell(i, j)

                # Add an attribute to identify the top cells.
                cell.is_topcell = True

                topcells.append(cell)

        # Store the indices of the surface cells so we can easily access them.
        self.topmost_indices = list(range(len(topcells)))

        # In the second pass, populate all the floors below, i.e. the cells
        # that are not on the surface.
        cells = []
        for top_cell in topcells:
            # Then, for each z-level that is below the top cell, we create
            # a cell (that is, we create the whole vertical column below
            # the current top cell.

            # Note that we create a cell by taking the current z_level as the
            # top of the cell (hence should be small that altitude of the top
            # cell and taking the previous z-level for the bottom of the cell.
            # Hence we exclude the lowest level from the looping.
            for i, z in enumerate(self.zlevels[1:]):
                if z <= top_cell.zl:
                    # Create a cell, whose vertical extent goes from the
                    # current level to the next one.
                    cell = Cell(top_cell.xl, top_cell.xh,
                            top_cell.yl, top_cell.yh,
                            self.zlevels[i - 1], z)
                    cells.append(cell)

        # We cast to numpy array, so that we can index also with lists.
        self.cells = np.array(topcells + cells)


    # TODO: Refactor: it would be better to have the 1D -> 2D indices
    # functionalities in the coarsener.
    def topmost_ind_to_2d_ind(self, ind):
        """ Given the index of a topmost cell in the list of cells, give the x
        and y index (in the 2D grid) which correspond to that cell.

        The goal of this method is to be able to find dsm cells that belong to
        a given topmost cell.

        Note that storing this as an attribute of each topmost cell would be
        memory costly, so we chose to compute it dynamically.

        Parameters
        ----------
        ind: int
            Index, in the 'cells' list of the topmost cell we are interested
            ind.

        Returns
        -------
        (int, int)
            x-y index (in the 2D version of the inversion grid) of the given
            cell. One can then use the get_fine_cells method of the coarsener
            to find the corresponding dsm cells.

        """
        ind_y = int(ind % self.dimy)
        ind_x = int((ind - ind_y) / self.dimy)

        return ((ind_x, ind_y))

    def fine_cells_from_topmost_ind(self, ind):
        """ Given the index of a topmost cell, give the fine cells that
        correspond to it.

        """
        # First convert to 2-D indexing.
        (ind_x, ind_y) = self.topmost_ind_to_2d_ind(ind)

        return self.coarsener.get_fine_cells(ind_x, ind_y)

    def ind_in_regular_grid(self, cell):
        """ Gives the a cell would have if it was in a regular 3D grid
        enclosing the irregular grid.

        The goal of this function is to be able to map inversion results to a
        regular 3D array, since most visualization softwares use that format.

        Parameters
        ----------
        cell: Cell

        Returns
        -------
        (i, j, k)
            Index of the cell in a regular grid that encloses the irregular
            one. The grid is chosen such that it just encloses the regular one.
            The grid doesn't care about individual cell resolutions.
            This is not much of a drawback since the only cells that dont have
            a standard resolution are on the borders fo the grid and will thus
            be clearly identifiable in a plot.

        """
