# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

from .SymbolicTablature import SymbolicTablature

# Regular imports
import os


class GuitarProTabs(SymbolicTablature):
    """
    Implements a wrapper for the GuitarPro dataset.
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                 profile=None, num_frames=None, split_notes=False, reset_data=False, store_data=True,
                 save_data=True, save_loc=None, seed=0, max_duration=0, augment_notes=False):
        """
        Initialize the dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See SymbolicTablature class...
        """

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames, split_notes,
                         reset_data, store_data, save_data, save_loc, seed, max_duration, augment_notes)

    def get_tracks(self, split):
        """
        Get the tracks associated with a dataset partition.

        Parameters
        ----------
        split : string
          Name of the partition from which to fetch tracks

        Returns
        ----------
        tracks : list of strings
          Names of tracks within the given partition
        """

        # Construct a path to GuitarPro jams directory
        jams_dir = os.path.join(self.base_dir, 'jams4')
        # Extract the names of all the files in the directory
        jams_paths = os.listdir(jams_dir)
        # Sort all of the tracks alphabetically
        jams_paths.sort()

        # Remove the JAMS file extension from the file names and take only those in split
        tracks = [os.path.splitext(path)[0] for path in jams_paths if split == path[0].lower()]

        return tracks

    def get_jams_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          GuitarSet track name

        Returns
        ----------
        jams_path : string
          Path to the JAMS file of the specified track
        """

        # Get the path to the annotations
        jams_path = os.path.join(self.base_dir, 'jams4', f'{track}.{tools.JAMS_EXT}')

        return jams_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. Currently, the splits
        are by first character (not case-sensitive) of track.

        Returns
        ----------
        splits : list of strings
          First characters of track names
        """

        splits = ['(', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                  'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                  'x', 'y', 'z']

        return splits

    @staticmethod
    def download(save_dir):
        """
        Download GuitarPro dataset to a specified location.
        TODO - we can probably set this up
        """

        return NotImplementedError
