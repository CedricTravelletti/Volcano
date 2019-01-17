import h5py
import numpy as np

# Number of observations.
N_OBS = 542

# Number of model cells.
N_MODEL = 179171

def load_niklas(path):
    """ Load Niklas data.

    Parameters
    ----------
    path: string
        Path to the HDF5.

    Returns
    -------
    Dict[F, dsm, coords]

    """
    dataset = h5py.File(path, 'r')

    # Pre-check dimensions.
    assert(np.allclose(dataset['F_land/data'].shape[0] / (N_OBS * N_MODEL), 1.0))
    assert(np.allclose(dataset['d_land'].shape[1] - 1, N_OBS))

    # The forward operator has been flattened into a list,
    # so we have to rebuild it.
    F = np.reshape(dataset['F_land/data'], (N_OBS, N_MODEL), order = 'F')

    # Measurement vector.
    # It has one element too much compared to what F expects,
    # hence we remove the first element, since it is 0.
    # (Maybe was included as reference point.)
    d = np.array(dataset['d_land'])[0, 1:]

    # Coordinates.
    xi = dataset['xi'][:, 0]
    yi = dataset['yi'][:, 0]
    zi = dataset['zi'][:, 0]

    # Have to subtract one to indices due to difference between matlab and python.
    ind = np.array(dataset['ind'], dtype=int) - 1

    # DSM
    # We have arrays of arrays, so we flatten to be one dimensional.
    dsm_x = np.ndarray.flatten(np.array(dataset['x']))
    dsm_y = np.ndarray.flatten(np.array(dataset['y']))
    dsm_z = np.array(dataset['z'])

    # Build a dsm matrix.
    dsm = []
    for i in range(dsm_x.size):
        for j in range(dsm_y.size):
            dsm.append([dsm_x[i], dsm_y[j], dsm_z[i, j]])

    dsm = np.array(dsm)

    # Build a coords matrix.
    coords = []
    for i in range(ind.shape[1]):
        # Read the indices in ind.
        ind_x = ind[2, i]
        ind_y = ind[1, i]
        ind_z = ind[0, i]

        # Get corresponding coords and add to list.
        coords.append([xi[ind_x], yi[ind_y], zi[ind_z]])

    # We put results in a numpy array for ease of use, it makes subsetting
    # easier.
    coords = np.array(coords)

    # Extract the data points locations.
    data_x = dataset['long'][0, :]
    data_y = dataset['lat'][0, :]
    data_z = dataset['h_true'][0, :]

    # Produce a list of coordinate tuples from that.
    # TODO: This might not be the most elegant way to store the coordinates of
    # the data points. Look into it.
    data_coords = []
    for x, y, z in zip(data_x, data_y, data_z):
        data_coords.append((x, y, z))

    return {'F': F, 'd': d, 'dsm': dsm, 'coords': coords,
            'data_coords': data_coords}
