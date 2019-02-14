import numpy as np

import plotly.offline as plt
import plotly.graph_objs as go


def plot(vals, coords, cmin=2150, cmax=2550.0, n_sample=0):
    """ Plot 3 dimensional scalar field.

    Parameters
    ----------
    vals: List[float]
        List of values to plot.
    x_coords: List[float]
        x-coordinate of the data points. Should have the same lenght as the
        list of values.
    y_coords: List[float]
    z_coords: List[float]
    n_sample: int
        If non zero, then will only plot n_sample randomly selected points from
        the dataset. Useful for visualizing heavy datasets.

    """
    dim = len(vals)

    if n_sample > 0:
        # Sample randomly and plot.
        plot_indices = np.random.random_integers(0, dim - 1, size=(n_sample))
    else:
        # Otherwise just use all indices.
        plot_indices = list(range(dim))

    plot_x = coords[plot_indices, 0]
    plot_y = coords[plot_indices, 1]
    plot_z = coords[plot_indices, 2]
    plot_vals = vals[plot_indices]

    myplot = go.Scatter3d(
            dict(
                    x=plot_x, y=plot_y, z=plot_z,
                    mode='markers',
                    marker=dict(size=3.2, opacity=1.0, color=plot_vals,
                    cmin=cmin, cmax=cmax,
                    colorscale='Jet', colorbar=dict(title='plot'))))

    layout = go.Layout()
    layout.scene.update(go.layout.Scene(aspectmode='data'))

    fig = go.Figure(data=[myplot], layout=layout)
    plt.plot(fig)


def plot_region(region_inds, vals, coords, cmin=2150, cmax=2550.0, n_sample=0):
    """ Same as above, but only plot a certain region. The region is defined by
    passing the indices of the cell in the region.

    Parameters
    ----------
    region_inds: 1D array-like
        Indices of the cells in the region.
    vals: List[float]
        List of values to plot.
    x_coords: List[float]
        x-coordinate of the data points. Should have the same lenght as the
        list of values.
    y_coords: List[float]
    z_coords: List[float]
    n_sample: int
        If non zero, then will only plot n_sample randomly selected points from
        the dataset. Useful for visualizing heavy datasets.

    """
    # Subset the corrresponding values.
    slice_vals = vals[region_inds]
    slice_coords = coords[region_inds, :]

    plot(slice_vals, slice_coords,cmin, cmax, n_sample)


def plot_z_slice(slice_height, vals, coords,cmin=2150, cmax=2550.0, n_sample=0):
    """ Same as above, but only plot as slice of fixed z coordinate.

    Parameters
    ----------
    slice_height: float or List[float]
        Value of the z coordinate along which to slice.
        If a list, then will plot several slices.
    vals: List[float]
        List of values to plot.
    x_coords: List[float]
        x-coordinate of the data points. Should have the same lenght as the
        list of values.
    y_coords: List[float]
    z_coords: List[float]
    n_sample: int
        If non zero, then will only plot n_sample randomly selected points from
        the dataset. Useful for visualizing heavy datasets.

    """
    # If list, then have to slice several times.
    if not isinstance(slice_height, list):
        # Get the indices of the cells in the slice.
        slice_inds = coords[:, 2] == slice_height
    else:
        # Create empyt boolean array, one hot encoding of cells we will plot.
        slice_inds = np.empty(coords.shape[0], dtype='bool')
        slice_inds[:] = False

        for h in slice_height:
            slice_inds = np.logical_or(slice_inds, coords[:, 2] == h)

    plot_region(slice_inds, vals, coords,cmin, cmax, n_sample)


def plot_x_slice(slice_x, vals, coords,cmin=2150, cmax=2550.0, n_sample=0):
    """ Same as above, but only plot as slice of fixed z coordinate.

    Parameters
    ----------
    slice_height: float or List[float]
        Value of the z coordinate along which to slice.
        If a list, then will plot several slices.
    vals: List[float]
        List of values to plot.
    x_coords: List[float]
        x-coordinate of the data points. Should have the same lenght as the
        list of values.
    y_coords: List[float]
    z_coords: List[float]
    n_sample: int
        If non zero, then will only plot n_sample randomly selected points from
        the dataset. Useful for visualizing heavy datasets.

    """
    # If list, then have to slice several times.
    if not isinstance(slice_x, list):
        # Get the indices of the cells in the slice.
        slice_inds = coords[:, 0] == slice_x
    else:
        # Create empyt boolean array, one hot encoding of cells we will plot.
        slice_inds = np.empty(coords.shape[0], dtype='bool')
        slice_inds[:] = False

        for h in slice_x:
            slice_inds = np.logical_or(slice_inds, coords[:, 0] == h)

    plot_region(slice_inds, vals, coords,cmin, cmax, n_sample)


def plot_y_slice(slice_y, vals, coords,cmin=2150, cmax=2550.0, n_sample=0):
    """ Same as above, but only plot as slice of fixed z coordinate.

    Parameters
    ----------
    slice_height: float or List[float]
        Value of the z coordinate along which to slice.
        If a list, then will plot several slices.
    vals: List[float]
        List of values to plot.
    x_coords: List[float]
        x-coordinate of the data points. Should have the same lenght as the
        list of values.
    y_coords: List[float]
    z_coords: List[float]
    n_sample: int
        If non zero, then will only plot n_sample randomly selected points from
        the dataset. Useful for visualizing heavy datasets.

    """
    # If list, then have to slice several times.
    if not isinstance(slice_y, list):
        # Get the indices of the cells in the slice.
        slice_inds = coords[:, 1] == slice_y
    else:
        # Create empyt boolean array, one hot encoding of cells we will plot.
        slice_inds = np.empty(coords.shape[0], dtype='bool')
        slice_inds[:] = False

        for h in slice_y:
            slice_inds = np.logical_or(slice_inds, coords[:, 1] == h)

    plot_region(slice_inds, vals, coords,cmin, cmax, n_sample)
