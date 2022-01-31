# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.evaluate import TablatureEvaluator

# Regular imports
import numpy as np

KEY_DP_ERRORS = 'duplicate_pitch_errors'


class DuplicatePitchError(TablatureEvaluator):
    """
    Implements an evaluator for counting the number of duplicate pitch predictions.
    """

    def __init__(self, profile, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See TablatureEvaluator class...
        """

        super().__init__(profile, key, save_dir, patterns, verbose)

    def evaluate(self, estimated, reference):
        """
        Evaluate a stacked multi pitch tablature estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        reference : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing number of duplicate pitch errors
        """

        # Collapse the stack by summation
        multipitch_est = np.sum(estimated, axis=-3)
        multipitch_ref = np.sum(reference, axis=-3)

        # Determine where the estimated and reference multipitch overlap
        valid_idcs = np.logical_and(multipitch_est, multipitch_ref)

        # Subtract the ground-truth pitch activations from the estimates
        duplicate_pitches = multipitch_est[valid_idcs] - multipitch_ref[valid_idcs]

        # Do not factor in cases where the estimated pitch count is smaller
        # than the reference - i.e., duplicated pitches in the ground-truth
        duplicate_pitches[duplicate_pitches < 0] = 0

        # Sum the duplicated pitches
        duplicate_pitch_errors = np.sum(duplicate_pitches)

        # Package the result into a dictionary
        results = {
            KEY_DP_ERRORS : duplicate_pitch_errors
        }

        return results