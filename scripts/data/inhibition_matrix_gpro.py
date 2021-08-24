# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import MelSpec

import amt_tools.tools as tools

# Regular imports
from tqdm import tqdm

import numpy as np
import os

# Private imports
import sys
sys.path.insert(0, '/')
from GuitarPro import GuitarProData

# Construct a path for saving the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'inhibition_matrix_no_string_3.npz')

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Determine the number of unique activations
num_activations = profile.get_num_dofs() * profile.num_pitches + profile.get_num_dofs()

# Create the data processing module (only because TranscriptionDataset needs it)
# TODO - it would be nice to not need this
data_proc = MelSpec(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_mels=192,
                    decibels=False,
                    center=False)

# Create a dataset using all of the tablature data
gpro_train = GuitarProData(base_dir=None,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           num_frames=2000,
                           data_proc=data_proc,
                           profile=profile,
                           save_data=False,
                           store_data=False,
                           max_duration=20,
                           augment_notes=True)

# Initialize the inhibition matrix with all zeros
inhibition_matrix = np.zeros((num_activations, num_activations))
# Initialize a matrix to hold the number of times a pair occurs at least once
valid_count = np.zeros(inhibition_matrix.shape)

# Loop through all of the tracks in the tablature dataset
#for i, track in enumerate(tqdm(gpro_train)):
max_iter = 10000000
for i in tqdm(range(max_iter)):
    track = gpro_train[gpro_train.rng.randint(0, len(gpro_train))]
    # Extract the tablature from the track data
    tablature = track[tools.KEY_TABLATURE]
    # Convert the tablature data to a stacked multi pitch array
    stacked_multi_pitch = tools.tablature_to_stacked_multi_pitch(tablature, profile)
    # Remove silent frames from the stacked multi pitch array
    stacked_multi_pitch = stacked_multi_pitch[..., np.sum(np.sum(stacked_multi_pitch, axis=-2), axis=-2) > 0]
    # Convert the stacked multi pitch array to logistic (unique string/fret) activations
    logistic = np.transpose(tools.stacked_multi_pitch_to_logistic(stacked_multi_pitch, profile, silence=True))

    # Determine which indices in the logistic activations correspond to string silence
    no_string_idcs = (profile.num_pitches + 1) * np.arange(profile.get_num_dofs())
    # Determine how many strings are inactive at each frame
    num_silent_strings = np.sum(logistic[..., no_string_idcs], axis=-1)
    # Obtain index pairs for silent string activations where more than N=2 strings are silent
    idx_pairs = np.meshgrid(np.where(num_silent_strings > 2)[0], no_string_idcs)
    # Ignore silent string activations unless N or more strings are active
    logistic[idx_pairs[0].flatten(), idx_pairs[1].flatten()] = 0

    # Count the number of frames each string/fret occurs in the tablature data
    single_occurrences = np.expand_dims(np.sum(logistic, axis=0), axis=0)
    # Sum the disjoint occurrences for each string/fret pair in the matrix
    disjoint_occurrences = np.repeat(single_occurrences, num_activations, axis=0) + \
                           np.repeat(single_occurrences.T, num_activations, axis=1)

    # Count the number of frames each string/fret pair occurs in the matrix
    co_occurrences = np.sum(np.matmul(np.reshape(logistic, (-1, num_activations, 1)),
                                      np.reshape(logistic, (-1, 1, num_activations))), axis=0)

    # Calculate the number of unique observations among the disjoint occurrences
    unique_occurrences = disjoint_occurrences - co_occurrences

    # Determine which string/fret combos have a non-zero count
    valid_idcs = np.repeat(single_occurrences != 0, num_activations, axis=0)
    # Determine the validity of modifying the indices of the inhibition matrix
    #   - Any row or column corresponding to an unseen string/fret combination is nullified
    valid_idcs = np.logical_and(valid_idcs, valid_idcs.T)
    # Add the index validity as +1 to the validity count
    valid_count += valid_idcs.astype(tools.INT64)

    # Calculate weight as the number of co-occurences over the number of unique observations
    weight = co_occurrences[valid_idcs] / unique_occurrences[valid_idcs]

    # Add the (nth-root boosted) weight (within [0, 1]) to the inhibition matrix (averaged at the end)
    inhibition_matrix[valid_idcs] += (weight ** 0.2)

    if (i + 1) % 1000 == 0:
        # Save the current inhibition matrix every 1000 tracks
        np.savez(save_path, inh=(1 - (inhibition_matrix / (valid_count + 1E-10))))

# Divide the summed track weights by the number of times the weight was valid
inhibition_matrix = inhibition_matrix / (valid_count + 1E-10)

# Subtract the weights from 1. We should end up with:
#   - Hard 0 for no penalty at all (i.e. self-association)
#   - Hard 1 for impossible combinations (i.e. dual-string)
#   - Somewhere in between for other correlations
inhibition_matrix = 1 - inhibition_matrix

# Save the inhibition matrix to disk
np.savez(save_path, inh=inhibition_matrix)
