import numpy as np

import plotly.offline as plt
import plotly.graph_objs as go


def plot(vals, x_coords, y_coords, z_coords, n_sample=0):
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

    plot_x = x_coords[plot_indices]
    plot_y = y_coords[plot_indices]
    plot_z = z_coords[plot_indices]
    plot_vals = vals[plot_indices]

    myplot = go.Scatter3d(
            dict(
                    x=plot_x, y=plot_y, z=plot_z,
                    mode='markers',
                    marker=dict(size=2, opacity=0.8, color=plot_vals,
                    colorscale='Jet', colorbar=dict(title='plot'))))

    plt.plot([myplot])


def plot_z_slice(slice_height, vals, x_coords, y_coords, z_coords, n_sample=0):
    """ Same as above, but only plot as slice of fixed z coordinate.

    Parameters
    ----------
    slice_height: float
        Value of the z coordinate along which to slice.
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
    # Get the indices of the cells in the slice.
    slice_inds = z_coords[:] == slice_height

    # Subset the corrresponding values.
    slice_vals = vals[slice_inds]
    slice_x_coords = x_coords[slice_inds]
    slice_y_coords = y_coords[slice_inds]
    slice_z_coords = z_coords[slice_inds]

    plot(slice_vals, slice_x_coords, slice_y_coords, slice_z_coords, n_sample=0)

def plot_region(region_inds, vals, x_coords, y_coords, z_coords, n_sample=0):
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
    slice_x_coords = x_coords[region_inds]
    slice_y_coords = y_coords[region_inds]
    slice_z_coords = z_coords[region_inds]

    plot(slice_vals, slice_x_coords, slice_y_coords, slice_z_coords, n_sample=0)
