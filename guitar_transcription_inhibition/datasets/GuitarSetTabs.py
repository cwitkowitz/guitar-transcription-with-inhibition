# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .SymbolicTablature import SymbolicTablature
from amt_tools.datasets import GuitarSet

import amt_tools.tools as tools

# Regular imports
import os


class GuitarSetTabs(GuitarSet, SymbolicTablature):
    """
    Implements a wrapper for GuitarSet symbolic tablature.
    """

    def __init__(self, **kwargs):
        """
        Initialize the dataset.

        Parameters
        ----------
        See SymbolicTablature class...
        """

        # Determine if the base directory argument was provided
        base_dir = kwargs.pop('base_dir', None)

        # Select a default base directory path if none was provided
        if base_dir is None:
            # Use the same naming scheme as regular GuitarSet
            base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, GuitarSet.dataset_name())

        # Update the argument in the collection
        kwargs.update({'base_dir' : base_dir})

        SymbolicTablature.__init__(self, **kwargs)

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

        # Load the track data using the symbolic tablature protocol
        data = SymbolicTablature.load(self, track)

        return data
