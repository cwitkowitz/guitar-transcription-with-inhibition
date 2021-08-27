# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
from math import inf

import numpy as np

EPSILON = 1E-10


def load_inhibition_matrix(save_path):
    """
    Helper function to load the NumPy Zip file containing the inhibition matrix.

    Parameters
    ----------
    save_path : string
      Path to inhibition matrix to load

    Returns
    ----------
    inhibition_matrix : ndarray (N x N)
      Matrix of inhibitory weights for string/fret pairs
      N - number of unique string/fret activations
    """

    # Load the inhibition matrix at the specified path
    inhibition_matrix = np.load(save_path)['inh']

    return inhibition_matrix


def trim_inhibition_matrix(inhibition_matrix, num_strings, num_pitches):
    """
    Helper function to trim away frets from the the inhibition matrix.

    Parameters
    ----------
    inhibition_matrix : ndarray (N x N)
      Matrix of inhibitory weights for string/fret pairs
      N - number of unique string/fret activations (untrimmed)
    num_strings : int
      Number of strings to expect in the inhibition matrix
    num_pitches : int
      Number of pitches per string to expect in the inhibition matrix

    Returns
    ----------
    inhibition_matrix : ndarray (M x M)
      Matrix of inhibitory weights for string/fret pairs
      M - number of unique string/fret activations (trimmed) (num_strings x num_pitches)
    """

    # Determine how many pitches were originally included in the matrix
    num_pitches_ = inhibition_matrix.shape[-1] // num_strings
    # Temporarily re-shape the matrix to be 4D
    inhibition_matrix = np.reshape(inhibition_matrix, (num_strings, num_pitches_,
                                                       num_strings, num_pitches_))
    # Throw away any extraneous frets
    inhibition_matrix = inhibition_matrix[:, :num_pitches, :, :num_pitches]
    # Calculate output dimensionality
    num_activations = num_strings * num_pitches
    # View the matrix as a square (2D) again
    inhibition_matrix = np.reshape(inhibition_matrix, (num_activations, num_activations))

    return inhibition_matrix


def plot_inhibition_matrix(inhibition_matrix, include_axes=True, labels=None, fig=None):
    """
    Static function for plotting an inhibition matrix heatmap.

    Parameters
    ----------
    inhibition_matrix : ndarray (N x N)
      Matrix of inhibitory weights for string/fret pairs
      N - number of unique string/fret activations
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

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    # Determine the number of unique activations
    num_activations = inhibition_matrix.shape[-1]

    # Obtain the complement of the matrix
    likelihood = 1 - inhibition_matrix

    # Check if a heatmap has already been plotted
    if len(ax.images):
        # Update the heatmap with the new data
        ax.images[0].set_data(likelihood)
    else:
        # Plot the inhibition matrix as a heatmap
        ax.imshow(likelihood, extent=[0, num_activations, num_activations, 0])

    # Add a title to the heatmap
    ax.set_title('Inhibition Matrix')

    if include_axes:
        if labels is None:
            # Default the string labels
            labels = tools.DEFAULT_GUITAR_LABELS
        # Determine the number of strings (implied by labels)
        num_strings = len(labels)
        # Obtain the indices where each string begins
        ticks = num_activations / num_strings * np.arange(num_strings)
        # Set the ticks as the start of the implied strings
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # Add tick labels to mark the strings
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        # Add gridlines to intersect the ticks
        ax.grid(c='black')
    else:
        # Hide the axes
        ax.axis('off')

    return fig


class InhibitionMatrixTrainer(object):
    """
    Implements the protocol for generating an inhibition matrix.
    """
    def __init__(self, profile, tablature_data, save_path, checkpoint_gap=1000, n_residual=100):
        """
        Initialize the internal state for the inhibition matrix training protocol.

        Parameters
        ----------
        profile : TablatureProfile (tools/instrument.py)
          Instructions for organizing tablature into logistic activations
        tablature_data : SymbolicTablature dataset
          Dataset for sampling symbolic tablature
        save_path : string
          Path to use when saving inhibition matrix
        checkpoint_gap : int
          Number of iterations between save checkpoints
        n_residual : int
          Number of iterations over which to average residual when checking stop condition
        """

        self.profile = profile
        self.tablature_data = tablature_data
        self.save_path = save_path
        self.checkpoint_gap = checkpoint_gap

        # Determine the number of unique activations
        self.num_activations = profile.get_num_dofs() * profile.num_pitches

        # Initialize pairwise weights with all zeros
        self.pairwise_weights = np.zeros((self.num_activations, self.num_activations))
        # Initialize a matrix to hold the number of times a pair occurs at least once
        self.valid_count = np.zeros(self.pairwise_weights.shape)

        # Initialize a counter for the current iteration
        self.current_iteration = 0
        # Initialize a residual buffer to use for feedback and stopping
        self.residual = np.array([inf] * n_residual)

    def train(self, residual_threshold=None):
        """
        Perform training steps for the inhibition matrix.

        Parameters
        ----------
        residual_threshold : float or None (optional)
          Residual threshold for stopping training
        """

        # Determine the total number of tracks in the dataset
        num_tracks = len(self.tablature_data)

        # Loop until a stop condition is met
        while True:
            if residual_threshold is None:
                # Stop when all tracks have been sampled
                stop_condition_met = self.current_iteration == num_tracks
            else:
                # Stop when the residual falls underneath the chosen threshold
                stop_condition_met = np.mean(self.residual) < residual_threshold

            if stop_condition_met:
                # Exit the while loop
                break

            if residual_threshold is None:
                # Sample tracks in-order
                sample_idx = self.current_iteration
            else:
                # Sample tracks randomly
                sample_idx = self.tablature_data.rng.randint(0, num_tracks)

            # Obtain activations for the sampled track
            logistic_activations = self.get_logistic_activations(sample_idx)

            # Update the matrix with the sampled activations
            self.step(logistic_activations)

            # Print the current training progress
            print(f'\rIteration : {self.current_iteration + 1} | Avg. Residual : {np.mean(self.residual)}', end='')

            # Determine if the current iteration is a checkpoint
            if (self.current_iteration + 1) % self.checkpoint_gap == 0:
                # Save the inhibition matrix
                self.save_current_matrix()

            # Increment the iteration counter
            self.current_iteration += 1

        # Print a newline character
        print()

        # Save the inhibition matrix one final time
        self.save_current_matrix()

    def get_logistic_activations(self, sample_idx):
        """
        Obtain frame-level activations for unique string/fret combinations.

        Parameters
        ----------
        sample_idx : int
          Sampled track index

        Returns
        ----------
        logistic_activations : ndarray (T x N)
          Array of tablature activations (e.g. string/fret combinations)
          T - number of frames
          N - number of unique string/fret activations
        """

        # Extract the sampled tablature from the dataset
        sampled_tabs = self.tablature_data[sample_idx][tools.KEY_TABLATURE]

        # Convert the tablature data to a stacked multi pitch array
        stacked_multi_pitch = tools.tablature_to_stacked_multi_pitch(sampled_tabs, self.profile)
        # Remove silent frames from the stacked multi pitch array
        stacked_multi_pitch = stacked_multi_pitch[..., np.sum(np.sum(stacked_multi_pitch, axis=-2), axis=-2) > 0]
        # Convert the stacked multi pitch array to logistic (unique string/fret) activations
        logistic_activations = np.transpose(tools.stacked_multi_pitch_to_logistic(stacked_multi_pitch, self.profile))

        return logistic_activations

    def step(self, logistic_activations):
        """
        Update the inhibition matrix components with a new tablature sample.

        Parameters
        ----------
        logistic_activations : ndarray (T x N)
          Array of tablature activations (e.g. string/fret combinations)
          T - number of frames
          N - number of unique string/fret activations
        """

        # Compute the matrix before the update
        previous_matrix = self.compute_current_matrix()

        # Count the number of frames each string/fret occurs in the tablature data
        single_occurrences = np.expand_dims(np.sum(logistic_activations, axis=0), axis=0)
        # Sum the disjoint occurrences for each string/fret pair in the matrix
        disjoint_occurrences = np.repeat(single_occurrences, self.num_activations, axis=0) + \
                               np.repeat(single_occurrences.T, self.num_activations, axis=1)

        # Count the number of frames each string/fret pair occurs in the matrix
        co_occurrences = np.sum(np.matmul(np.reshape(logistic_activations, (-1, self.num_activations, 1)),
                                          np.reshape(logistic_activations, (-1, 1, self.num_activations))), axis=0)

        # Calculate the number of unique observations among the disjoint occurrences
        unique_occurrences = disjoint_occurrences - co_occurrences

        # Determine which string/fret combos have a non-zero count
        valid_idcs = np.repeat(single_occurrences != 0, self.num_activations, axis=0)
        # Determine the validity of modifying the indices of the pairwise weights
        #   - Any row or column corresponding to an unseen string/fret combination is nullified
        valid_idcs = np.logical_and(valid_idcs, valid_idcs.T)
        # Add the index validity as +1 to the validity count
        self.valid_count += valid_idcs.astype(tools.INT64)

        # Calculate weight as the number of co-occurences over the number of unique observations (IoU)
        iteration_weight = co_occurrences[valid_idcs] / unique_occurrences[valid_idcs]

        # Add the (nth-root boosted) weight (within [0, 1]) to the pairwise weights
        self.pairwise_weights[valid_idcs] += (iteration_weight ** 0.2) # 5th root

        # Compute the residual between the current and previous matrix
        residual = np.sum(np.abs(self.compute_current_matrix() - previous_matrix))

        # Shift the residual buffer
        self.residual = np.roll(self.residual, shift=-1)
        # Add the new residual to the buffer
        self.residual[-1] = residual

    def compute_current_matrix(self):
        """
        Compute the current inhibition matrix using the
        current pairwise weights and the current validity count.

        Returns
        ----------
        inhibition_matrix : ndarray (N x N)
          Matrix of inhibitory weights for string/fret pairs
          N - number of unique string/fret activations
        """

        # Divide the pairwise weights by the number of times the weights were valid
        inhibition_matrix = self.pairwise_weights / (self.valid_count + EPSILON)

        # Subtract the weights from 1. We should end up with:
        #   - Hard 0 for no penalty at all (i.e. self-association)
        #   - Hard 1 for impossible combinations (i.e. dual-string)
        #   - Somewhere in between for other correlations
        inhibition_matrix = 1 - inhibition_matrix

        return inhibition_matrix

    def save_current_matrix(self):
        """
        Helper function to save a NumPy Zip file containing the inhibition matrix.
        """

        # Compute the current inhibition matrix
        inhibition_matrix = self.compute_current_matrix()

        # Save the inhibition matrix to disk
        np.savez(self.save_path, inh=inhibition_matrix)
