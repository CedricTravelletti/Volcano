""" This submodule contains functions for building artificial irregular grids
(topographies) when building synthetic volcanoes.

It can also generate data measurement site on the surface of the topography
(sites placed at random) and compute the forward operator associated to the
topography/data sites.

"""
import numpy as np
from multiprocessing import Pool, RawArray
from volcapy.niklas.banerjee import banerjee


# Gravitational constant.
G = 6.67e-6       #Transformation factor to get result in mGal

# =====================
# Multiprocessing Stuff
# =====================
# A global dictionary storing the variables passed from the initializer.
var_dict = {}
def init_worker(F, F_shape, coords, coords_shape, data, data_shape, meta):
    var_dict['F'] = F
    var_dict['F_shape'] = F_shape
    var_dict['coords'] = coords
    var_dict['coords_shape'] = coords_shape
    var_dict['data_coords'] = data
    var_dict['data_coords_shape'] = data_shape
    var_dict['meta'] = meta

def _worker_func(i):
    F_np = np.frombuffer(var_dict['F']).reshape(var_dict['F_shape'])
    coords_np = np.frombuffer(var_dict['coords']).reshape(var_dict['coords_shape'])
    data_np = np.frombuffer(var_dict['data_coords']).reshape(var_dict['data_coords_shape'])

    res_x = var_dict['meta']['res_x']
    res_y = var_dict['meta']['res_y']
    res_z = var_dict['meta']['res_z']

    cell = coords_np[i, :]
    tmp = _compute_forward_column(
            cell, res_x, res_y, res_z, data_np)
    F_np[:, i] = tmp
    return None

def build_cube(nr_x, res_x, nr_y, res_y, nr_z, res_z):
    """ Builds a regular gridded cube.

    Parameters
    ----------
    nr_x: int
        Number of cells in x-dimension.
    res_x: float
        Size of cell along x_dimension.
    nr_y: int
    res_y: float
    nr_z: int
    res_z: float

    Returns
    -------
    ndarray
        Array of size n_cells * 3.
        Contains the coordinates of the center of each cell.

    """
    # Make sure first cell starts at [0, 0, 0]
    orig_x = res_x / 2.0
    orig_y = res_y / 2.0
    orig_z = res_z / 2.0

    coords = []

    current_x = orig_x
    for i in range(nr_x):
        current_y = orig_y
        for j in range(nr_y):
            current_z = orig_z
            for k in range(nr_z):
                coords.append([current_x, current_y, current_z])
                current_z += res_z
            current_y += res_y
        current_x += res_x
    return np.array(coords)

def compute_forward(coords, res_x, res_y, res_z, data_coords, n_procs):
    """ Compute the forward operator associated to a given topography/irregular
    grid. In the end, it only need a list of cells.

    Parameters
    ----------
    coords: ndarray
        Cells centroid coordinates, size n_cell * n_dims.
    res_x: float
        Length of a cell in x-direction (meters).
    res_y_float
    res_z: float
    data_coords: ndarray
        List of data measurements coordinates, size n_data * n_dims.
    n_procs: int
        Number of processes to use to parallelize computation.

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    n_cells = coords.shape[0]
    n_dim_coords = coords.shape[1]
    n_data = data_coords.shape[0]
    n_dim_data = data_coords.shape[1]
    F_shape = (n_data, n_cells)
    data_shape = (n_data, n_dim_data)
    coords_shape = (n_cells, n_dim_coords)

    meta = {'res_x': res_x, 'res_y': res_y, 'res_z': res_z, }

    # ----------------------------
    # Prepare the parallelization.
    # ----------------------------
    F_shared_buffer = RawArray('d', n_data * n_cells)
    # Wrap as a numpy array so we can easily manipulates its data.
    F_np = np.frombuffer(F_shared_buffer).reshape(F_shape)
    # Copy data to our shared array.
    np.copyto(F_np, np.zeros(F_shape))

    coords_shared_buffer = RawArray('d', n_cells * n_dim_coords)
    # Wrap as a numpy array so we can easily manipulates its data.
    coords_np = np.frombuffer(coords_shared_buffer).reshape(coords_shape)
    # Copy data to our shared array.
    np.copyto(coords_np, coords)

    # Same with data, so noesnt need to be copied along processes.
    data_coords_shared_buffer = RawArray('d', n_data * n_dim_data)
    # Wrap as a numpy array so we can easily manipulates its data.
    data_coords_np = np.frombuffer(data_coords_shared_buffer).reshape(data_shape)
    # Copy data to our shared array.
    np.copyto(data_coords_np, data_coords)


    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of
    # each worker.
    # (Because X_shape is not a shared variable,
    # it will be copied to each
    # child process.)
    with Pool(processes=n_procs, initializer=init_worker,
            initargs=(F_shared_buffer, F_shape,
                    coords_shared_buffer, coords_shape,
                    data_coords_shared_buffer, data_shape, meta)) as pool:
        result = pool.map(_worker_func, range(coords_shape[0]))

    return F_np

def _compute_forward_column(cell, res_x, res_y, res_z, data_coords):
    """ Helper function for parallelizing the computation of the forward.

    """
    n_data = data_coords.shape[0]
    F_column = np.zeros(n_data)

    for j, data in enumerate(data_coords):
        # Compute cell endpoints.
        xh = cell[0] + res_x / 2.0
        xl = cell[0] - res_x / 2.0
        yh = cell[1] + res_y / 2.0
        yl = cell[1] - res_y / 2.0
        zh = cell[2] + res_z / 2.0
        zl = cell[2] - res_z / 2.0

        F_column[j] = G * banerjee(
                xh, xl, yh, yl, zh, zl,
                data[0],  data[1],  data[2])
    return F_column

def generate_regular_surface_datapoints(
        xl, xh, nx, yl, yh, ny, zl, zh, nz,
        offset):
    """ Put regularly spaced measurement points on the surface of a
    cube.
    Note that there will always be measurement sites at the endpoints of the
    cube.
    We need an offset because measerements cannot be directly on the endpoints
    of a cell because of division by zero in the Bannerjee formula.

    Parameters
    ----------
    xl: float
        Lower x-coordinate of the cube.
    xh: float
        Higher x-coordinate of the cube.
    nx: int
        Number of measurments in x-dimension.
    yl: float
    yh: float
    ny: int
    zl: float
    zh: float
    nz: int
    offset: float
        Displace the measurements sites by an offset outside of the cube to
        avoid division by zero.

    Returns
    -------
    ndarray
        Coordinates of the measurement sites, size n_data * n_dims.

    """
    data_coords = []

    # Bottom x surface.
    for y in np.linspace(yl - offset, yh + offset, ny):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([xl - offset, y, z])

    # Top x surface.
    for y in np.linspace(yl - offset, yh + offset, ny):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([xh + offset, y, z])

    # Bottom y surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([x, yl - offset, z])

    # Top y surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([x, yh + offset, z])

    # Bottom z surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for y in np.linspace(yl - offset, yh + offset, ny):
            data_coords.append([x, y, zl - offset])

    # Top z surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for y in np.linspace(yl - offset, yh + offset, ny):
            data_coords.append([x, y, zh + offset])
    return np.array(data_coords)

def build_cone_in_cube(coords):
    """ Given a cubic grid, turn it into a cone.
    That is, given a list of cells that form a full cubic grid, place a cone in
    the cube and mark the cells belonging to the cone.

    Parameters
    ----------
    coords: ndarray
        Array of size n_cells * 3.
        Contains the coordinates of the center of each cell.

    Returns
    -------
    ndarray
        1 dimensional array containing indices of cells belonging to the cone.

    """
    # Center in the x-y plane.
    x_center = np.mean(coords[:, 0])
    y_center = np.mean(coords[:, 1])

    x_radius = (np.max(coords[:, 0]) - np.min(coords[:, 0])) / 2.0
    y_radius = (np.max(coords[:, 1]) - np.min(coords[:, 1])) / 2.0

    # Take as radius of the cone the mean of the two radiuses.
    R = (x_radius + y_radius) / 2.0

    # z-extent.
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])

    # Cone condition.
    cone_inds = np.where(
            (coords[:, 0] - x_center)**2 + (coords[:, 1] - y_center)**2
            <= R**2 * (1 - (coords[:, 2] - z_min) / (z_max - z_min))**2)[0]

    return cone_inds

def build_conic_volcano(coords, nx, ny, nz):
    """ Build an artificial volcano having a conic shape.

    Parameters
    ----------
    coords: ndarray
        Array of size n_cells * 3.
        Contains the coordinates of the center of each cell.
    nx: int
        Number of cells along x-dimension.
    ny: int
    nz: int

    Returns
    -------
    ndarray
        1 dimensional array containing indices of cells belonging to the cone.

    """
    cone_inds = build_cone_in_cube(coords)

    # Get the indices of the surfcace.
    tmp = np.zeros(coords[:, 0].shape)
    tmp[cone_inds] = 1
    tmp = np.reshape(tmp, (nx, ny, nz))

    # For eax x-y point, find highest z and mark it.
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            for k in range(tmp.shape[2]):
                # Soon as we detect a zero (soon as we transition out of the
                # volcano), we mark the last encountered cell (along
                # z-direction) as a surface cell.
                if tmp[i, j , k] == 0:
                    if tmp[i , j, k - 1] == 1:
                        tmp[i , j, k - 1] = 2
                        break
    
    # Reshape to normal grid.
    tmp = np.reshape(tmp, (-1))
    surface_inds = np.where(tmp[:] == 2)[0]

    return np.array(cone_inds), np.array(surface_inds)

def meshgrid_from_coords(coords, nx, ny, nz):
    """ Turns a list of coordinates (in regular grid)
    into a meshgrid.

    """
    return np.reshape(coords, (nx, ny, nz, 3))

def coords_from_meshgrid(meshgrid):
    """ Inverse operation of the above.

    """
    return np.reshape(meshgrid, (-1, 3))
