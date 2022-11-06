# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import load_inhibition_matrix, \
                                                       trim_inhibition_matrix
from guitar_transcription_inhibition.models import LogisticTablatureEstimator
from guitar_transcription_inhibition.datasets import GuitarSetTabs

import amt_tools.tools as tools

# Regular imports
import numpy as np
import torch
import os


def get_observation_statistics(tablature_dataset, profile, silence_activations=False):
    """
    Count the number of times each string/fret pair occurs within a symbolic tablature dataset.

    Parameters
    ----------
    tablature_dataset : SymbolicTablature dataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations
    silence_activations : bool
      Whether to keep track of statistics for string silence

    Returns
    ----------
    total_frames : int
      Total number of frames encountered in the dataset
    occurrences : ndarray (S x C)
      Total number of occurrences for each string/fret
      S - number of degrees of freedom (strings)
      C - number of classes
    """

    # Determine the number of distinct activations in the tablature
    num_activations = profile.get_num_dofs() * (profile.num_pitches + int(silence_activations))

    # Initialize a counter for the total number of frames
    # and the total number of occurrences of each string/fret
    total_frames, occurrences = 0, np.zeros(num_activations)

    # Loop through the symbolic tablature dataset
    for track_data in tablature_dataset:
        # Extract the tablature
        tablature = track_data[tools.KEY_TABLATURE]
        # Convert the tablature to logistic activations
        logistic = tools.tablature_to_logistic(tablature, profile, silence_activations)

        # Update the frame count
        total_frames += logistic.shape[-1]
        # Update the number of occurrences for each string/fret
        occurrences += np.sum(logistic, axis=-1)

    # Reshape the occurrences so they can be indexed by string
    occurrences = np.reshape(occurrences, (profile.get_num_dofs(), -1))

    if silence_activations:
        # Move the silence classes to the final index
        occurrences = np.roll(occurrences, -1, axis=-1)

    return total_frames, occurrences


def compute_balanced_class_weighting(tablature_dataset, profile, silence_activations=False):
    """
    Obtain the weighting necessary to balance the tablature classes.

    Parameters
    ----------
    tablature_dataset : SymbolicTablature dataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations
    silence_activations : bool
      Whether to keep track of statistics for string silence

    Returns
    ----------
    balanced_weighting : ndarray (S x C)
      Weights for classes such that they will be balanced
      S - number of degrees of freedom (strings)
      C - number of classes
    """

    # Determine the number of classes in the tablature
    num_classes = profile.num_pitches + int(silence_activations)

    # Obtain the total number of frames and occurrences of each string/fret
    total_frames, occurrences = get_observation_statistics(tablature_dataset, profile, silence_activations)

    # Compute the balanced weighting
    balanced_weighting = total_frames / (num_classes * occurrences)
    # Replace infinite weights with zeros (for classes which did not occur at all)
    balanced_weighting[balanced_weighting == np.inf] = 0

    return balanced_weighting


def compute_dataset_inhibition_loss(tablature_dataset, inhibition_matrix, profile, silence_activations=False):
    """
    Compute the inhibition loss over an entire symbolic
    tablature dataset, given a pre-existing inhibition matrix.

    Parameters
    ----------
    tablature_dataset : TranscriptionDataset
      Dataset containing symbolic tablature
    inhibition_matrix : tensor (N x N)
      Matrix of inhibitory weights for string/fret pairs
      N - number of unique string/fret activations
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations
    silence_activations : bool
      Whether to keep track of statistics for string silence

    Returns
    ----------
    total_loss : float
      Average inhibition loss over each track in the dataset
    """

    # Initialize a list to keep track of each track's loss
    losses = []

    # Convert the inhibition matrix to a Tensor
    inhibition_matrix = tools.array_to_tensor(inhibition_matrix)

    # Loop through the symbolic tablature dataset
    for track_data in tablature_dataset:
        # Extract the tablature
        tablature = track_data[tools.KEY_TABLATURE]
        # Convert the tablature to logistic activations
        logistic = tools.tablature_to_logistic(tablature, profile, silence_activations)
        # Switch the dimensions and convert the activations to a Tensor
        logistic = torch.Tensor(logistic.T)

        # Compute the inhibition loss for the track
        inhibition_loss = LogisticTablatureEstimator.calculate_inhibition_loss(logistic, inhibition_matrix)

        # Add the loss to the tracked list
        losses += [inhibition_loss]

    # Average the loss across all tracks
    total_loss = torch.mean(torch.Tensor(losses))

    return total_loss


def count_dataset_duplicate_pitches(tablature_dataset, profile):
    """
    Compute the average number of duplicate pitches (frame-level) over an entire symbolic tablature dataset.

    Parameters
    ----------
    tablature_dataset : TranscriptionDataset
      Dataset containing symbolic tablature
    profile : TablatureProfile (tools/instrument.py)
      Instructions for organizing tablature into logistic activations

    Returns
    ----------
    total_duplicates : float
      Average number of duplicate pitches across each track in the dataset
    """

    # Initialize a list to keep track of each track's count
    duplicates = []

    # Loop through the symbolic tablature dataset
    for track_data in tablature_dataset:
        # Extract the tablature
        tablature = track_data[tools.KEY_TABLATURE]
        # Convert from tablature format to stacked multi pitch format
        stacked_multipitch = tools.tablature_to_stacked_multi_pitch(tablature, profile)
        # Collapse the stack by summation
        multipitch = np.sum(stacked_multipitch, axis=-3)
        # Subtract one for single occurrences
        multipitch[multipitch > 0] = multipitch[multipitch > 0] - 1

        # Sum the remaining (duplicated) pitch counts and add to the tracked list
        duplicates += [np.sum(multipitch)]

    # Average the loss across all tracks
    total_duplicates = np.mean(duplicates)

    return total_duplicates


if __name__ == '__main__':
    # Processing parameters
    sample_rate = 22050
    hop_length = 512

    # Construct a path for loading the inhibition matrix
    matrix_path = os.path.join('..', '..', 'generated', 'matrices', '<MATRIX>.npz')

    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Extract tablature parameters
    num_strings = profile.get_num_dofs()
    num_pitches = profile.num_pitches

    # All data/features cached here
    gset_cache = os.path.join('..', '..', 'generated', 'data')

    # Initialize GuitarSet tablature to compute statistics
    tablature_dataset = GuitarSetTabs(base_dir=None,
                                      hop_length=hop_length,
                                      sample_rate=sample_rate,
                                      profile=profile,
                                      num_frames=None,
                                      save_loc=gset_cache)

    # Nicer printing for arrays
    np.set_printoptions(suppress=True)

    # Count the number of occurrences of each string/fret pair
    total_frames, occurrences = get_observation_statistics(tablature_dataset, profile, True)

    print(f'Total Frames : {total_frames}')
    print(f'String/Fret Occurrences : \n{occurrences}')

    # Obtain a balanced class weighting for the dataset
    balanced_weighting = compute_balanced_class_weighting(tablature_dataset, profile, True)

    print(f'Balanced Weighting : \n{balanced_weighting}')

    # Load the specified inhibition matrix
    inhibition_matrix = load_inhibition_matrix(matrix_path)

    # Trim the inhibition matrix to match the chosen profile
    inhibition_matrix = trim_inhibition_matrix(inhibition_matrix, num_strings, num_pitches, silence_activations=True)

    # Compute the inhibition loss on the full dataset
    inhibition_loss = compute_dataset_inhibition_loss(tablature_dataset, inhibition_matrix, profile, True)

    print(f'Inhibition Loss : {inhibition_loss}')

    # Compute the average number of duplicate pitches on the full dataset
    duplicate_pitches = count_dataset_duplicate_pitches(tablature_dataset, profile)

    print(f'Avg. Duplicate Pitches : {duplicate_pitches}')
