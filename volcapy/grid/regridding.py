""" Lower grid resolution.
This is part of the effort to implement multi-resolution GP regression.

"""
import numpy as np
from volcapy.grid.sparsifier import Sparsifier


def irregular_regrid_single_step(cells_coords, step_resolution):
    """ Lower resolution by one step, in an irregular fashion.

    Principle of the algorithm is the following:
      We start by picking a cell at random in the list.
      We then aggregate all cells that are within *step_resolution* of this
      cell. All the aggregated cells then form on big cell, and the
      corrsponding original cells are then removed from the list.

      We then pick another cell in the list and proceed in the same way till
      the list is empty.
    The correspondence between cell indices in the coarse grid and cell indices
    in the fine grid is kept in a list that is returned by the algorithm. The
    i-th element of this list contains a list of cell indices (in the fine
    grid) that have been aggregated to form cell i in the coarse grid.

    Parameters
    ----------
    cell_coords: ndarray, n_cells*n_dims
        Array of cell coordinates.
    step_resolution: float
        Given a starting cell, we will aggregate all cells that are within this
        distance (in the infinity norm) of the starting cell.

    Returns
    -------
    coarse_cells_coords: ndarray
        Coordinates of the cells of the coarse grid (same format as the input).
    coarse_to_fine_inds: List[List[int]]
        Correspondence between the grids. Element i contains list of indices in
        fine grid that have been aggregated to form cell i in coarse grid.

    """
    sparsifier = Sparsifier(cells_coords, infty_metric=True)
    coarse_cells_coords = []
    coarse_to_fine_inds = []

    # We keep a list of indices of unused cells. Each time we use a cell, we
    # will put the corresponding element to -1 and wont visit it anymore.
    # Use ndarray to allow indexing by lists.
    candidate_cells = np.array(list(range(cells_coords.shape[0])))

    for candidate_ind in candidate_cells:
        # Skip if already used.
        if candidate_ind < 0:
            continue
        # Get indices of cells within distance.
        candidate_cell = cells_coords[candidate_ind]
        neighbor_inds = sparsifier.get_cells_ind_with_radius(
                candidate_cell, step_resolution)

        # New cell coords is average of constituent cells.
        coarse_cells_coords.append(
                np.mean(cells_coords[neighbor_inds], axis=0))

        # Update mapping and remove used cells from candidates.
        coarse_to_fine_inds.append(neighbor_inds)
        candidate_cells[neighbor_inds] = -1

    coarse_cells_coords = np.array(coarse_cells_coords)
    coarse_cells_coords = np.asfortranarray(coarse_cells_coords)
    return (coarse_cells_coords, coarse_to_fine_inds)

def regrid_forward(F, coarse_to_fine_inds):
    """ Adapt the forward to the new grid.

    """
    n_cells = len(coarse_to_fine_inds)
    F_new = np.zeros((F.shape[0], n_cells), dtype=np.float32)

    for i, fine_inds in enumerate(coarse_to_fine_inds):
        new_column = np.sum(F[:, fine_inds], axis=1)
        F_new[:, i] = new_column

    return F_new
