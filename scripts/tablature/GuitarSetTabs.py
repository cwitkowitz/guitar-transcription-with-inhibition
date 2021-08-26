# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet

import amt_tools.tools as tools

from .SymbolicTablature import SymbolicTablature

# Regular imports
import os


class GuitarSetTabs(SymbolicTablature):
    """
    Implements a wrapper for the GuitarSet dataset tablature.
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

        # Select a default base directory path if none was provided
        if base_dir is None:
            # Use the same naming scheme as regular GuitarSet
            base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, GuitarSet.dataset_name())

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

        # Construct a path to GuitarSet's JAMS directory
        jams_dir = os.path.join(self.base_dir, 'annotation')
        # Extract the names of all the files in the directory
        jams_paths = os.listdir(jams_dir)
        # Sort all of the tracks alphabetically
        jams_paths.sort()

        # Remove the JAMS file extension from the file names
        tracks = [os.path.splitext(path)[0] for path in jams_paths]

        # Determine where the split starts within the sorted tracks
        split_start = int(split) * 60
        # Slice the appropriate tracks
        tracks = tracks[split_start : split_start + 60]

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
        jams_path = os.path.join(self.base_dir, 'annotation', f'{track}.{tools.JAMS_EXT}')

        return jams_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. Currently, the splits are by player,
        but we could equally do genre, key, etc., as long as get_tracks() is adapted.

        Returns
        ----------
        splits : list of strings
          Player codes listed at beginning of file names
        """

        # Same splits as regular GuitarSet
        splits = GuitarSet.available_splits()

        return splits

    @staticmethod
    def download(save_dir):
        """
        Download GuitarSet to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of GuitarSet
        """

        # Same steps as regular GuitarSet
        GuitarSet.download(save_dir)
