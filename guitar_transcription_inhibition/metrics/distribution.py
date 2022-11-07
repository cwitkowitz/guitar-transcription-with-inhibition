# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import trim_inhibition_matrix
from guitar_transcription_inhibition.models import LogisticTablatureEstimator
from amt_tools.evaluate import TablatureEvaluator, LossWrapper

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

    def evaluate(self, estimated, reference):
        """
        Evaluate a stacked multi pitch tablature estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x T)
          Array of class membership for multiple degrees of freedom (e.g. strings)
          S - number of strings or degrees of freedom
          T - number of frames
        reference : ndarray (S x T)
          Array of class membership for multiple degrees of freedom (e.g. strings)
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing number of duplicate pitch errors
        """

        # Convert from tablature format to logistic activations format (with no silence class)
        logistic_est = tools.tablature_to_logistic(estimated, self.profile, silence=False)
        logistic_ref = tools.tablature_to_logistic(reference, self.profile, silence=False)

        # Compute the number of false alarm errors
        false_alarm_errors = np.sum(np.logical_and(logistic_est, np.logical_not(logistic_ref)), dtype=tools.FLOAT)

        # Convert the tablature into stacked multi pitch representations
        stacked_multi_pitch_est = tools.tablature_to_stacked_multi_pitch(estimated, self.profile)
        stacked_multi_pitch_ref = tools.tablature_to_stacked_multi_pitch(reference, self.profile)

        # Collapse the stack by summation
        multi_pitch_est = np.sum(stacked_multi_pitch_est, axis=-3)
        multi_pitch_ref = np.sum(stacked_multi_pitch_ref, axis=-3)

        # Determine where the estimated and reference multipitch overlap
        valid_idcs = np.logical_and(multi_pitch_est, multi_pitch_ref)

        # Subtract the ground-truth pitch activations from the estimates
        duplicate_pitches = multi_pitch_est[valid_idcs] - multi_pitch_ref[valid_idcs]

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


class InhibitionLoss(TablatureEvaluator, LossWrapper):
    """
    Implements an evaluator for measuring the inhibition loss of final predictions.
    """

    def __init__(self, profile, matrices, silence_activations, unpack_key=None,
                 results_key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See TablatureEvaluator class for others...

        matrices : dict (str -> inhibition_matrix : tensor (N x N))
          Dictionary containing all inhibition matrices to evaluate
        silence_activations : bool
          Whether the silent string is explicitly modeled as an activation
        """

        super().__init__(profile, unpack_key, results_key, save_dir, patterns, verbose)

        self.matrices = tools.dict_to_tensor(matrices)
        self.silence_activations = silence_activations

    def evaluate(self, estimated, reference=None):
        """
        Calculate the inhibition loss(es) for the estimated tablature.

        Parameters
        ----------
        estimated : ndarray (S x T)
          Array of class membership for multiple degrees of freedom (e.g. strings)
          S - number of strings or degrees of freedom
          T - number of frames
        reference : irrelevant

        Returns
        ----------
        results : dict
          Dictionary containing number of duplicate pitch errors
        """

        # Extract the final tablature predictions and convert from tablature to logistic format
        logistic_est = tools.tablature_to_logistic(estimated, self.profile, silence=self.silence_activations)
        # Switch the frame and activation dimension, add a batch dimension, and convert to tensor
        logistic_est = tools.array_to_tensor(np.expand_dims(logistic_est.T, axis=0))

        # Initialize a new dictionary to hold various inhibition losses
        inhibition_losses = dict()

        # Loop through each inhibition matrix provided
        for matrix_key in self.matrices.keys():
            # Unpack the matrix from the dictionary and trim to match the specified profile
            inhibition_matrix = trim_inhibition_matrix(self.matrices[matrix_key],
                                                       num_strings=self.profile.get_num_dofs(),
                                                       num_pitches=self.profile.num_pitches,
                                                       silence_activations=self.silence_activations)
            # Make sure the inhibition matrix is on the appropriate device
            inhibition_matrix = inhibition_matrix.to(logistic_est.device)
            # Compute the inhibition loss on the final predictions
            inhibition_loss = LogisticTablatureEstimator.calculate_inhibition_loss(logistic_est, inhibition_matrix)
            # Add the inhibition loss to the dictionary
            inhibition_losses[matrix_key] = inhibition_loss.item()

        # Package the result into a dictionary
        results = {
            KEY_INH_SCORE : inhibition_losses
        }

        return results
