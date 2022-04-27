# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ..models import LogisticTablatureEstimator
from amt_tools.evaluate import TablatureEvaluator

import amt_tools.tools as tools

# Regular imports
import numpy as np

KEY_FA_ERRORS = 'false_alarm_errors'
KEY_DP_ERRORS = 'duplicate_pitch_errors'
KEY_INH_SCORE = 'inhibition_score'


class FalseAlarmErrors(TablatureEvaluator):
    """
    Implements an evaluator for counting the number of false alarm errors.
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

        # Treat the stacked multi pitch arrays as logistic activations (with no silence class)
        # TODO - add to pre_proc?
        logistic_est = tools.stacked_multi_pitch_to_logistic(estimated, self.profile, False)
        logistic_ref = tools.stacked_multi_pitch_to_logistic(reference, self.profile, False)

        # Compute the number of false alarm errors
        false_alarm_errors = np.sum(np.logical_and(logistic_est, np.logical_not(logistic_ref)))

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
            KEY_FA_ERRORS : false_alarm_errors,
            KEY_DP_ERRORS : duplicate_pitch_errors
        }

        return results


class InhibitionLoss(TablatureEvaluator):
    """
    Implements an evaluator for measuring the inhibition loss of final predictions.
    """

    def __init__(self, profile, matrices, silence_activations, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See TablatureEvaluator class for others...

        matrices : dict (str -> inhibition_matrix : tensor (N x N))
          Dictionary containing all inhibition matrices to evaluate
        """

        super().__init__(profile, key, save_dir, patterns, verbose)

        self.matrices = matrices
        self.silence_activations = silence_activations

    def evaluate(self, estimated, reference):
        """
        Calculate the inhibition losses for the estimated tablature.

        Parameters
        ----------
        estimated : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        reference (unused) : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing number of duplicate pitch errors
        """

        # Extract the final tablature predictions
        # TODO - add to pre_proc?
        logistic_est = tools.stacked_multi_pitch_to_logistic(estimated, self.profile, self.silence_activations)
        # Switch the frame and activation dimension, and add a batch dimension
        logistic_est = np.expand_dims(logistic_est.T, axis=0)
        # Convert the predictions to Tensor format
        logistic_est = tools.array_to_tensor(logistic_est)

        # Initialize a new dictionary to hold various inhibition scores
        inhibition_losses = dict()

        # Loop through each inhibition matrix provided
        for matrix_key in self.matrices.keys():
            # Unpack the matrix from the dictionary
            inhibition_matrix = self.matrices[matrix_key]
            # Add the predictions to the appropriate device
            logistic_est = logistic_est.to(inhibition_matrix.device)
            # Compute the inhibition score on the final predictions
            inhibition_loss = LogisticTablatureEstimator.calculate_inhibition_loss(logistic_est, inhibition_matrix)
            # Add the score to the dictionary
            inhibition_losses[matrix_key] = inhibition_loss.item()

        # Package the result into a dictionary
        results = {
            KEY_INH_SCORE : inhibition_losses
        }

        return results
