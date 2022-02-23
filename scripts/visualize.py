# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from inhibition.inhibition_matrix_utils import InhibitionMatrixTrainer, plot_inhibition_matrix
from models.tabcnn_variants import TabCNNRecurrent
from amt_tools.models import TabCNN

import amt_tools.tools as tools

# Regular imports
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


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


def visualize(model, tablature_dataset, save_dir, config=[0, 0, 1], lhood_select=2, i=None):
    """
    Visualize the activations of a network on a dataset.

    Parameters
    ----------
    model : TabCNN, TabCNNRecurrent, or TabCNNLogistic
      Model which produces tablature output
    tablature_dataset : TranscriptionDataset
      Dataset containing symbolic tablature
    save_dir : string
      Directory under which to save images
    config : list of int or bool
      Whether to visualize raw, softmax, and final activations, respectively
    lhood_select : int
      Selection of which activations (0 - raw, 1 - softmax, 2 - final)
      to use for computing pairwise likelihoods of predictions
    i : int
      Current iteration for directory organization
    """

    if i is not None:
        # Add an additional folder for the checkpoint
        save_dir = os.path.join(save_dir, f'checkpoint-{i}')

    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Indicate whether the raw activations will include a silent string activation
    silent_class = True if isinstance(model, TabCNN) or \
                           isinstance(model, TabCNNRecurrent) else model.silence_activations

    # Initialize a matrix trainer for visualizing the pairwise likelihood of activations
    trainer = InhibitionMatrixTrainer(model.profile, silent_string=silent_class, power=1)

    # Determine the parameters of the tablature
    num_strings = model.profile.get_num_dofs()
    num_classes = model.profile.num_pitches + int(silent_class)

    # Loop through all tracks in the validation set
    for n, track_data in enumerate(tablature_dataset):
        # Extract the track name to use for save paths
        track_name = track_data[tools.KEY_TRACK]

        # Convert the track data to Tensors and add a batch dimension
        track_data = tools.dict_unsqueeze(tools.dict_to_tensor(track_data))
        # Run the track through the model without completing pre-processing steps
        raw_predictions = model(model.pre_proc(track_data)[tools.KEY_FEATS])
        # Throw away tracked gradients so the predictions can be copied
        raw_predictions = tools.dict_detach(raw_predictions)

        # Copy the predictions and remove the batch dimension
        raw_activations = tools.dict_squeeze(deepcopy(raw_predictions), dim=0)
        # Extract the tablature and switch the time and string/fret dimensions
        raw_activations = raw_activations[tools.KEY_TABLATURE].T

        # Reshape the activations according to the separate softmax groups
        smax_activations = raw_activations.view(num_strings, num_classes, -1)
        # Apply the softmax function to the activations for each group
        smax_activations = torch.softmax(smax_activations, dim=-2)
        # Return the activations to their original shape
        smax_activations = smax_activations.view(num_strings * num_classes, -1)

        # Convert the raw activations and softmax activations to NumPy arrays
        raw_activations = tools.tensor_to_array(raw_activations)
        smax_activations = tools.tensor_to_array(smax_activations)

        # Package the predictions and ground-truth together for post-processing
        output = {tools.KEY_OUTPUT: deepcopy(raw_predictions),
                  tools.KEY_TABLATURE: track_data[tools.KEY_TABLATURE].to(model.device)}
        # Obtain final predictions and remove the batch dimension
        final_activations = tools.dict_squeeze(model.post_proc(output), dim=0)
        # Extract the tablature activations and convert to NumPy array
        final_activations = tools.tensor_to_array(final_activations[tools.KEY_TABLATURE])
        # Convert the final tablature predictions to logistic activations
        final_activations = tools.tablature_to_logistic(final_activations, model.profile, silent_class)

        # Process the chosen activations for the pairwise likelihoods
        if lhood_select == 0:
            trainer.step(raw_activations)
        elif lhood_select == 1:
            trainer.step(smax_activations)
        else:
            trainer.step(final_activations)

        # Determine the times associate with each frame
        times = np.arange(raw_activations.shape[-1]) * tablature_dataset.hop_length / tablature_dataset.sample_rate

        if config[0]:
            # Construct a save path for the raw activations
            raw_path = os.path.join(save_dir, f'{track_name}-raw.jpg')
            # Create a figure for plotting
            fig = plt.figure(figsize=(16, 8))
            # Plot the raw activations and return the figure
            fig = plot_logistic_activations(raw_activations, times=times, fig=fig)
            # Bump up the aspect ratio
            fig.gca().set_aspect(0.09)
            # Save the figure to the specified path
            fig.savefig(raw_path, bbox_inches='tight')
            # Close the figure
            plt.close(fig)

        if config[1]:
            # Construct a save path for the softmax activations
            smax_path = os.path.join(save_dir, f'{track_name}-softmax.jpg')
            # Create a figure for plotting
            fig = plt.figure(figsize=(16, 8))
            # Plot the softmax activations and return the figure
            fig = plot_logistic_activations(smax_activations, times=times, v_bounds=[0, 1], fig=fig)
            # Bump up the aspect ratio
            fig.gca().set_aspect(0.09)
            # Save the figure to the specified path
            fig.savefig(smax_path, bbox_inches='tight')
            # Close the figure
            plt.close(fig)

        if config[2]:
            # Construct a save path for the final activations
            final_path = os.path.join(save_dir, f'{track_name}-final.jpg')
            # Create a figure for plotting
            fig = plt.figure(figsize=(16, 8))
            # Plot the final activations and return the figure
            fig = plot_logistic_activations(final_activations, times=times, v_bounds=[0, 1], fig=fig)
            # Bump up the aspect ratio
            fig.gca().set_aspect(0.09)
            # Save the figure to the specified path
            fig.savefig(final_path, bbox_inches='tight')
            # Close the figure
            plt.close(fig)

    # Compute the pairwise activations
    pairwise_activations = trainer.compute_current_matrix()

    # Construct a save path for the pairwise weights
    inhibition_path = os.path.join(save_dir, f'_pairwise_likelihood.jpg')
    # Make sure the save directory exists
    os.makedirs(os.path.dirname(inhibition_path), exist_ok=True)

    # Create a figure for plotting
    fig = plt.figure(figsize=(10, 10))
    # Plot the inhibition matrix and return the figure
    fig = plot_inhibition_matrix(pairwise_activations, v_bounds=[0, 1], fig=fig)
    # Save the figure to the specified path
    fig.savefig(inhibition_path, bbox_inches='tight')
    # Close the figure
    plt.close(fig)
