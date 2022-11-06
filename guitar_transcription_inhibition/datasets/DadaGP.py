# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .SymbolicTablature import SymbolicTablature

import amt_tools.tools as tools

# Regular imports
import os


class DadaGP(SymbolicTablature):
    """
    Implements a wrapper for the DadaGP dataset.
    """

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

        # Obtain a list of all the JAMS files
        jams_files = os.listdir(os.path.join(self.base_dir, 'jams'))

        # Determine which sub-splits to reference for the split
        sub_splits = self.val_sub_splits() if split == 'val' else self.train_sub_splits()

        # Reduce the list of files to those in the split and remove the JAMS extension
        tracks = [os.path.splitext(track)[0] for track in jams_files if self.get_first_track_char(track) in sub_splits]

        return tracks

    def get_jams_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          DadaGP track name

        Returns
        ----------
        jams_path : string
          Path to the JAMS file of the specified track
        """

        # Get the path to the annotations
        jams_path = os.path.join(self.base_dir, 'jams', f'{track}.{tools.JAMS_EXT}')

        return jams_path

    @staticmethod
    def get_first_track_char(track):
        """
        Obtain the first valid character for a track name.

        Parameters
        ----------
        track : string
          DadaGP track name

        Returns
        ----------
        first_char : string
          First valid character within the track name
        """

        # Get the list of available sub-splits (sub-directories)
        valid_chars = DadaGP.available_sub_splits()

        # Initialize a character index
        char_idx = 0

        # Default the first valid character
        first_char = None

        # Loop until a valid character is found or the end of the track name
        while first_char is None and char_idx < len(track):
            # Get the character at the current index and capitalize
            curr_char = track[char_idx].upper()

            if curr_char in valid_chars:
                # Set the first character is it is valid
                first_char = curr_char

            # Increment the character index
            char_idx += 1

        return first_char

    @staticmethod
    def available_sub_splits():
        """
        Obtain a list of all sub-splits (alphanumeric).

        Returns
        ----------
        sub_splits : list of strings
          Sub-directories within the base directory
        """

        sub_splits = ['1', '2', '3', '4', '5', '6', '7', '8',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                      'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'Z']

        return sub_splits

    @staticmethod
    def train_sub_splits():
        """
        Obtain a list of sub-splits (alphanumeric) that belong to the training set.

        Returns
        ----------
        sub_splits : list of strings
          First character of JAMS files
        """

        # Compliment of the validation sub-splits w.r.t. all available splits
        sub_splits = [ss for ss in DadaGP.available_sub_splits() if ss not in DadaGP.val_sub_splits()]

        return sub_splits

    @staticmethod
    def val_sub_splits():
        """
        Obtain a list of sub-splits (alphanumeric) that belong to the validation set.

        Returns
        ----------
        sub_splits : list of strings
          First character of JAMS files
        """

        sub_splits = ['T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        return sub_splits

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. Currently, the
        splits are by training and validation partitions.

        Returns
        ----------
        splits : list of strings
          Tags for training and validation split
        """

        splits = ['train', 'val']

        return splits

    @staticmethod
    def download(save_dir):
        """
        Download DadaGP dataset to a specified location.
        """

        assert False, 'Please see \'https://github.com/dada-bots/dadaGP\' for steps to acquire DadaGP...'
