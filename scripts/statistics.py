# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

import amt_tools.tools as tools

from inhibition.inhibition_matrix_utils import load_inhibition_matrix, trim_inhibition_matrix
from models.tablature_layers import LogisticTablatureEstimator

# Regular imports
import numpy as np
import torch
import os


def get_observation_statistics(tablature_dataset, profile, string_silence=False):
    """
    Count the number of times each string/fret occurs in a symbolic tablature dataset.

    Parameters
    ----------
    tablature_dataset : SymbolicTablature dataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations
    string_silence : bool
      Whether to keep track of statistics for string silence

    Returns
    ----------
    total_num_frames : int
      Total number of frames encountered in the dataset
    num_observations : ndarray (S x C)
      Total number of occurences for each string/fret
      S - number of degrees of freedom (strings)
      C - number of classes
    """

    # Determine the number of classes and the number of activations in the tablature
    num_classes = profile.num_pitches + int(string_silence)
    num_activations = profile.get_num_dofs() * num_classes

    # Initialize a counter for the total number of frames
    # and the total number of occurrences of each string/fret
    total_frames, occurrences = 0, np.zeros(num_activations)

    # Loop through the symbolic tablature dataset
    for track_data in tablature_dataset:
        # Extract the tablature
        tablature = track_data[tools.KEY_TABLATURE]
        # Convert the tablature to logistic activations
        logistic = tools.tablature_to_logistic(tablature, profile, string_silence)

        # Update the frame count
        total_frames += logistic.shape[-1]
        # Update the number of occurrences for each string/fret
        occurrences += np.sum(logistic, axis=-1)

    # Reshape the occurrences so they can be indexed by string
    occurrences = np.reshape(occurrences, (profile.get_num_dofs(), -1))

    if string_silence:
        # Move the silence classes to the final index
        occurrences = np.roll(occurrences, -1, axis=-1)

    return total_frames, occurrences


def compute_balanced_class_weighting(tablature_dataset, profile, string_silence=False):
    """
    Obtain a weighting to balance the tablature classes.

    Parameters
    ----------
    tablature_dataset : SymbolicTablature dataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations
    string_silence : bool
      Whether to keep track of statistics for string silence

    Returns
    ----------
    balanced_weighting : ndarray (S x C)
      Weights for classes such that they will be balanced
      S - number of degrees of freedom (strings)
      C - number of classes
    """

    # Determine the number of classes in the tablature
    num_classes = profile.num_pitches + int(string_silence)

    # Obtain the total number of frames and occurrences of each string/fret
    total_frames, occurrences = get_observation_statistics(tablature_dataset, profile, string_silence)

    # Compute the balanced weighting
    balanced_weighting = total_frames / (num_classes * occurrences)
    # Replace infinite weights with zeros (for classes which did not occur at all)
    balanced_weighting[balanced_weighting == np.inf] = 0

    return balanced_weighting


def compute_alternative_class_weighting(tablature_dataset, profile):
    """
    Obtain a weighting to transfer the energy
    of the silent classes to the positive classes.

    Parameters
    ----------
    tablature_dataset : SymbolicTablature dataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations

    Returns
    ----------
    class_weighting : ndarray (S x C)
      Weights for classes such that the positive
      classes receive the energy of the silence classes
      S - number of degrees of freedom (strings)
      C - number of classes
    """

    # Obtain the total number of frames and occurrences of each string/fret
    total_frames, occurrences = get_observation_statistics(tablature_dataset, profile, True)

    # Determine the percentage of classes where each string/fret occurs
    class_weighting = occurrences / (total_frames + 1E-10)

    # Obtain a reference to the weight of the silence classes
    silence_weights = class_weighting[:, -1]

    # Transfer the weight of the silence classes to the non-silent classes
    class_weighting[:, :-1] *= np.expand_dims(silence_weights / (1 - silence_weights), axis=-1)
    # Reverse the weight of the silence classes
    class_weighting[:, -1] = 1 - silence_weights

    return class_weighting


def compute_dataset_inhibition_loss(tablature_dataset, inhibition_matrix, profile, string_silence=False):
    """
    Compute the inhibition loss over an entire symbolic
    tablature dataset, given a pre-existing inhibition matrix.

    Parameters
    ----------
    tablature_dataset : TranscriptionDataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations
    string_silence : bool
      Whether to keep track of statistics for string silence

    Returns
    ----------
    total_num_frames : int
      Total number of frames encountered in the dataset
    num_observations : ndarray (S x C)
      Total number of occurences for each string/fret
      S - number of degrees of freedom (strings)
      C - number of classes
    """

    # Initialize a list to keep track of each track's loss
    losses = []

    # Convert the inhibition matrix to a Tensor
    inhibition_matrix = tools.array_to_tensor(inhibition_matrix)

    # Loop through the symbolic tablature dataset
    for track_data in tablature_dataset:
        # Extract the tablature
        tablature = track_data[tools.KEY_TABLATURE]
        # Convert the tablature to logistic activations and switch the dimensions
        logistic = tools.tablature_to_logistic(tablature, profile, string_silence)
        # Switch the dimensions and convert the activations to a Tensor
        logistic = torch.Tensor(logistic.T)

        # Compute the inhibition loss for the track
        inhibition_loss = LogisticTablatureEstimator.calculate_inhibition_loss(logistic, inhibition_matrix)

        # Add the loss to the tracked list
        losses += [inhibition_loss]

    # Average the loss across all tracks
    total_loss = torch.mean(torch.Tensor(losses))

    return total_loss


if __name__ == '__main__':
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Processing parameters
    sample_rate = 22050
    hop_length = 512
    dim_in = 192

    # Create the data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # All data/features cached here
    gset_cache = os.path.join('..', 'generated', 'data')

    # Initialize GuitarSet to compute statistics
    tablature_dataset = GuitarSet(base_dir=None,
                                  hop_length=hop_length,
                                  sample_rate=sample_rate,
                                  data_proc=data_proc,
                                  profile=profile,
                                  save_loc=gset_cache)

    # Nicer printing for arrays
    np.set_printoptions(suppress=True)

    # Count the number of occurrences of each string/fret
    total_frames, occurrences = get_observation_statistics(tablature_dataset, profile, True)

    print(f'Total Frames : {total_frames}')
    print(f'String/Fret Occurrences : \n{occurrences}')

    # Obtain a balanced class weighting for the dataset
    balanced_weighting = compute_balanced_class_weighting(tablature_dataset, profile, True)

    print(f'Balanced Weighting : \n{balanced_weighting}')

    # Obtain an alternative class weighting for the dataset
    alternative_weighting = compute_alternative_class_weighting(tablature_dataset, profile)

    print(f'Alternative Weighting : \n{alternative_weighting}')

    matrix_path = 'path/to/matrix'

    # Load the inhibition matrix
    inhibition_matrix = load_inhibition_matrix(matrix_path)

    # Extract tablature parameters
    num_strings = profile.get_num_dofs()
    num_pitches = profile.num_pitches

    # Trim the inhibition matrix to match the chosen profile
    inhibition_matrix = trim_inhibition_matrix(inhibition_matrix, num_strings, num_pitches, silent_string=True)

    # Compute the inhibition loss on the full dataset
    inhibition_loss = compute_dataset_inhibition_loss(tablature_dataset, inhibition_matrix, profile, True)

    print(f'Inhibition Loss : {inhibition_loss}')
