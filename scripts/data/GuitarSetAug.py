# Author: Frank Cwitkowitz <frank@chordify.net>

# My imports
from amt_tools.datasets import GuitarSet

import amt_tools.tools as tools

# Private imports
import sys
sys.path.insert(0, '/home/rockstar/Desktop/guitar-transcription/scripts/data')
from augment_tablature import *

# Regular imports
from copy import deepcopy

import numpy as np
import librosa
import jams
import os


class GuitarSetAug(GuitarSet):
    """
    Implements a wrapper for the GuitarSet dataset with augmented tablature.
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                 profile=None, num_frames=None, split_notes=False, reset_data=False, save_loc=None,
                 seed=0, augment_notes=False):
        """
        Initialize the dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See GuitarSet class for others...
        augment_notes : bool
          Whether or not to augment the data at the note level
        """

        # Select a default base directory path if none was provided
        if base_dir is None:
            base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, GuitarSet.dataset_name())

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile,
                         num_frames, split_notes, reset_data, False, False, save_loc, seed)

        self.augment_notes = augment_notes

    def __getitem__(self, index):
        """
        Extending this method to remove the hop length from the ground-truth when sampling.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict
          Dictionary containing the features and ground-truth data for the sampled track
        """

        # Call the main __getitem__() function
        data = super().__getitem__(index)

        # Remove sampling rate - it can cause problems if it is not an ndarray. Sample rate
        # should be able to be inferred from the dataset object, if no warnings are thrown
        if tools.query_dict(data, tools.KEY_HOP):
            data.pop(tools.KEY_HOP)

        return data

    def get_track_data(self, track_id, frame_start=None, frame_length=None, max_attempts=10):
        """
        Get the features and ground truth for a track within a time interval.

        Parameters
        ----------
        track_id : string
          Name of track data to fetch
        frame_start : int
          Frame with which to begin the slice
        frame_length : int
          Number of frames to take for the slice
        max_attempts : int
          Maximum number of attempts to sample non-zero frames

        Returns
        ----------
        data : dict
          Dictionary with each entry sliced for the random or provided interval
        """

        if self.store_data:
            # Copy the track's ground-truth data into a local dictionary
            data = deepcopy(self.data[track_id])
        else:
            # Load the track's ground-truth
            data = self.load(track_id)

        # Check to see if a specific frame length was given
        if frame_length is None:
            # If not, and this Dataset object has a frame length, use it
            if self.num_frames is not None:
                frame_length = self.num_frames
            # Otherwise, we assume the whole track is desired and perform no further actions
            else:
                return data

        # If a specific starting frame was not provided
        while frame_start is None or max_attempts:
            # Determine the last frame at which we can start
            sampling_end_point = data[tools.KEY_TABLATURE].shape[-1] - frame_length
            # Sample a group of frames randomly
            frame_start = self.rng.randint(0, sampling_end_point) if sampling_end_point > 0 else 0

            # Determine where the sample of frames should end
            frame_end = frame_start + frame_length

            # Check if non-silent frames were sampled
            if np.sum(data[tools.KEY_TABLATURE][..., frame_start : frame_end] != -1):
                # No more attempts are needed
                max_attempts = 0
            else:
                # Decrement the number of attempts
                max_attempts -= 1

        # Slice the remaining dictionary entries
        data = tools.slice_track(data, frame_start, frame_end, skip=[tools.KEY_FS, tools.KEY_HOP])

        return data

    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          Name of the track to load

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """

        # Initialize a new dictionary if there is no saved data
        data = dict()

        # Add the track ID to the dictionary
        data[tools.KEY_TRACK] = track

        # Construct the path to the track's JAMS data
        jams_path = self.get_jams_path(track)

        # Open up the JAMS data
        jam = jams.load(jams_path)

        # Get the total duration of the file
        duration = tools.extract_duration_jams(jam)

        # Load the notes by string from the JAMS file
        stacked_notes = tools.extract_stacked_notes_jams(jam)

        if self.augment_notes:
            # Randomly drop 50% of notes
            stacked_notes = random_notes_drop(stacked_notes, 0.5, self.rng)
            # Perform capo augmentation 50% of the time
            if self.rng.rand() <= 0.5:
                # Place a capo randomly, such that all notes can still be played relative to the capo fret
                stacked_notes = random_capo_move(stacked_notes, self.profile, self.rng)
            # Randomly shift onsets and offsets with a standard deviation of 50 ms
            stacked_notes = random_onset_offset_shift(stacked_notes, 0.050, self.rng)

        # Get the times for the start of each frame
        times = tools.get_frame_times(duration, self.sample_rate, self.hop_length)

        # Convert the string-wise notes into a stacked multi pitch array
        stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

        # Convert the stacked multi pitch array into a single representation
        data[tools.KEY_MULTIPITCH] = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

        # Convert the stacked multi pitch array into tablature
        data[tools.KEY_TABLATURE] = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

        # Add the sampling rate and the hop length to the data
        data[tools.KEY_FS], data[tools.KEY_HOP] = self.sample_rate, self.hop_length

        return data
