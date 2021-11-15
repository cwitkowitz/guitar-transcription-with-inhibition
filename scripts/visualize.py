# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np


def plot_logistic_activations(activations, times=None, v_bounds=None, include_axes=True, labels=None, fig=None):
    """
    Static function for plotting a heatmap for logistic activations.

    Parameters
    ----------
    activations : tensor (N x T)
      Array of distinct activations (e.g. string/fret combinations)
      N - number of individual activations
      T - number of frames
    times : ndarray or None (Optional)
      Times corresponding to frames
    v_bounds : list, ndarray, or None (Optional)
      Boundaries for plotting the heatmap
    include_axes : bool
      Whether to include the axis in the plot
    labels : list of string or None (Optional)
      Labels for the strings
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the inhibition matrix
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = tools.initialize_figure(interactive=False)

    if v_bounds is None:
        v_bounds = [None, None]

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    # Determine the number of unique activations and frames
    num_activations, num_frames = activations.shape

    # Check if a heatmap has already been plotted
    if len(ax.images):
        # Update the heatmap with the new data
        ax.images[0].set_data(activations)
    else:
        extent = [0, num_frames if times is None else times[-1], num_activations, 0]
        # Plot the inhibition matrix as a heatmap
        ax.imshow(activations, extent=extent, vmin=v_bounds[0], vmax=v_bounds[1])

    if include_axes:
        if labels is None:
            # Default the string labels
            labels = tools.DEFAULT_GUITAR_LABELS
        # Determine the number of strings (implied by labels)
        num_strings = len(labels)
        # Obtain the indices where each string begins
        ticks = num_activations / num_strings * np.arange(num_strings)
        # Set the ticks as the start of the implied strings
        ax.set_yticks(ticks)
        # Add tick labels to mark the strings
        ax.set_yticklabels(labels)

        if times is None:
            ax.set_xlabel('Frame')
        else:
            ax.set_xlabel('Time (s)')

        # Add gridlines to intersect the ticks
        ax.grid(c='black')
    else:
        # Hide the axes
        ax.axis('off')

    return fig
