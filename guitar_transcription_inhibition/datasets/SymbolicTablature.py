# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import TranscriptionDataset

import amt_tools.tools as tools

# Regular imports
from copy import deepcopy

import numpy as np
import jams


class SymbolicTablature(TranscriptionDataset):
    """
    Implements a wrapper for any dataset with symbolic tablature stored in JAMS format.
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                 profile=None, num_frames=None, split_notes=False, reset_data=False, store_data=True,
                 save_data=True, save_loc=None, seed=0, max_duration=0):
        """
        Initialize the dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionDataset class for others...
        max_duration : float
          Maximum duration in minutes a JAMS file can represent. Files longer than this will be randomly
          sliced to fit the maximum. Depending on the amount of RAM available, converting a very long
          duration to frames can cause a crash. Set max_duration=0 to disable, i.e., any duration is fine.
        """

        self.max_duration = 60 * max_duration # Convert to seconds

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames,
                         -1, split_notes, reset_data, store_data, save_data, save_loc, seed)

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
        Get the ground truth for a track within a time interval, skipping feature extraction.

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

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = super().load(track)

        # If the track data is being instantiated, it will not have the tablature key
        if tools.KEY_TABLATURE not in data.keys():
            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Open up the JAMS data
            jam = jams.load(jams_path)

            # Get the total duration of the file
            duration = tools.extract_duration_jams(jam)

            # Check if a maximum duration is set and if it has been exceeded
            if self.max_duration and duration > self.max_duration:
                # Sample a starting point randomly
                sec_start = self.rng.rand() * (duration - self.max_duration)

                # Slice the jam data
                jam = jam.slice(sec_start, sec_start + self.max_duration)

                # Re-extract the duration
                duration = tools.extract_duration_jams(jam)

            # Load the notes by string from the JAMS file
            stacked_notes = tools.extract_stacked_notes_jams(jam)

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

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Save the data as a NumPy zip file
                keys = (tools.KEY_FS, tools.KEY_HOP, tools.KEY_TABLATURE, tools.KEY_MULTIPITCH)
                tools.save_pack_npz(gt_path, keys, data[tools.KEY_FS], data[tools.KEY_HOP],
                                    data[tools.KEY_TABLATURE], data[tools.KEY_MULTIPITCH])

        return data
