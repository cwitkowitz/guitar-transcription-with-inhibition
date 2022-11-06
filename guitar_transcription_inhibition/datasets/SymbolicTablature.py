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
    Implements a wrapper for a dataset with symbolic tablature stored in JAMS format.
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=22050, profile=None, num_frames=None,
                 split_notes=False, reset_data=False, store_data=False, save_data=True, save_loc=None, seed=0,
                 max_duration=0, max_sample_attempts=1):
        """
        Initialize a symbolic tablature dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionDataset class for others...

        max_duration : float
          Maximum duration of data (in minutes) that can be extracted from a JAMS file. If more data is
          available, a random slice with duration equal to the maximum will be extracted. Depending on
          the amount of RAM available, representing a very large amount of data as frames can cause a
          crash. Set max_duration=0 to disable, i.e., to fully process files with any duration.
        max_sample_attempts : int
          Maximum number of attempts to sample a segment that is not entirely non-zero frames
        """

        self.max_duration = 60 * max_duration # Convert to seconds
        self.max_sample_attempts = max_sample_attempts

        super().__init__(base_dir, splits, hop_length, sample_rate, None, profile, num_frames,
                         None, split_notes, reset_data, store_data, save_data, save_loc, seed)

    # TODO - delete if unnecessary
    """
    def __getitem__(self, index):
        "/""
        Extension of parent method which additionally removes hop length from ground-truth when sampling.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict
          Dictionary containing the features and ground-truth data for the sampled track
        "/""

        # Call the main __getitem__() function
        data = super().__getitem__(index)

        if tools.query_dict(data, tools.KEY_HOP):
            # Remove hop length from ground-truth
            data.pop(tools.KEY_HOP)

        return data
    """

    def get_track_data(self, track_id, frame_start=None, frame_length=None):
        """
        Get the ground truth for a track within an interval, skipping feature extraction.

        Parameters
        ----------
        track_id : string
          Name of track data to fetch
        frame_start : int
          Frame with which to begin the slice
        frame_length : int
          Number of frames to take for the slice

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

        # Determine the last frame at which an interval can begin
        sampling_end_point = data[tools.KEY_TABLATURE].shape[-1] - frame_length

        # Keep track of the amount of sampling attempts
        attempts_remaining = self.max_sample_attempts

        # If a specific starting frame was not provided, sample one randomly
        while frame_start is None or attempts_remaining:
            # Sample a starting frame for the interval randomly
            frame_start = self.rng.randint(0, sampling_end_point) if sampling_end_point > 0 else 0

            # Determine where the sample of frames should end
            frame_end = frame_start + frame_length

            # Check if non-silent frames were sampled
            if np.sum(data[tools.KEY_TABLATURE][..., frame_start : frame_end] != -1):
                # No more attempts are needed
                attempts_remaining = 0
            else:
                # Decrement the number of attempts
                attempts_remaining -= 1

        # Slice all ground-truth to the sampled interval
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
        if not tools.query_dict(data, tools.KEY_TABLATURE):
            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the JAMS data
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

            # Represent the string-wise notes as a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Convert the stacked multi pitch array into tablature
            tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Convert the stacked multi pitch array into a single representation
            multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

            # Add all relevant ground-truth to the dictionary
            data.update({tools.KEY_FS : self.sample_rate,
                         # TODO - delete if unnecessary
                         #tools.KEY_HOP : self.hop_length,
                         tools.KEY_TABLATURE : tablature,
                         tools.KEY_MULTIPITCH : multi_pitch})

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Save the data as a NumPy zip file
                tools.save_dict_npz(gt_path, data)

        return data

    def get_jams_path(self, track):
        """
        This method should be implemented by child classes.

        Parameters
        ----------
        track : string
          Dataset track name
        """

        return NotImplementedError
