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
