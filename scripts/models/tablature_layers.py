# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import TranscriptionModel, SoftmaxGroups, LogisticBank
import amt_tools.tools as tools

from inhibition.inhibition_matrix import load_inhibition_matrix, trim_inhibition_matrix

# Regular imports
import numpy as np
import torch


class TablatureEstimator(TranscriptionModel):
    def __init__(self, dim_in, profile, device='cpu'):
        """
        Initialize the tablature layer as a TranscriptionModel.

        Parameters
        ----------
        See TranscriptionModel class...
        """

        # Call TranscriptionModel.__init__()
        super().__init__(dim_in, profile, 1, 1, 1, device)

        # Initialize a null tablature output layer
        self.tablature_layer = None

    def pre_proc(self, batch):
        """
        Treat the ground-truth multipitch as the features, extracting the
        ground-truth multipitch from the ground-truth tablature if necessary.

        Parameters
        ----------
        batch : dict
          Dictionary containing ground-truth tablature (and potentially multipitch) for a group of tracks

        Returns
        ----------
        batch : dict
          Dictionary containing ground-truth multipitch as features
        """

        # If there are no pre-existing features, add in the ground-truth multipitch
        if not tools.query_dict(batch, tools.KEY_FEATS):
            if tools.query_dict(batch, tools.KEY_MULTIPITCH):
                # Extract the multipitch from the ground-truth
                multipitch = batch[tools.KEY_MULTIPITCH]
            elif tools.query_dict(batch, tools.KEY_TABLATURE):
                # Obtain the tablature from the ground-truth and make sure it consists of integers
                tablature = tools.unpack_dict(batch, tools.KEY_TABLATURE).to(torch.int32)
                # Convert the tablature to stacked multipitch arrays
                stacked_multipitch = tools.tablature_to_stacked_multi_pitch(tablature, self.profile)
                # Collapse into a single multipitch array
                multipitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multipitch)
            else:
                # This will cause an error
                multipitch = None

            # Flip the dimensions of the multipitch and add as features to the dictionary
            batch[tools.KEY_FEATS] = multipitch.transpose(-1, -2)

        return batch

    def forward(self, embeddings):
        """
        Perform the main processing steps for the tablature layer.

        Parameters
        ----------
        embeddings : Tensor (B x T x F)
          Feature embeddings for a batch of tracks,
          B - batch size
          T - number of frames
          F - dimensionality of features

        Returns
        ----------
        output : dict w/ tablature Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames)
          O - tablature output dimensionality
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Compute the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings.float())

        return output

    def post_proc(self, batch):
        """
        Calculate tablature loss.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature and potentially loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Unpack the loss if it exists
        loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

        if loss is None:
            # Create a new dictionary to hold the loss
            loss = dict()

        # Check to see if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Calculate the loss and add it to the total
            tablature_loss = self.tablature_layer.get_loss(tablature_est, batch[tools.KEY_TABLATURE])
            # Add the tablature loss to the tracked loss dictionary
            loss[tools.KEY_LOSS_TABS] = tablature_loss
            # Add the tablature loss to the total loss
            total_loss += tablature_loss

        # Determine if loss is being tracked
        if total_loss:
            # Add the loss to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        return output


class ClassicTablatureEstimator(TablatureEstimator):
    """
    A model wrapper for the Softmax groups output layer from
    TabCNN (http://archives.ismir.net/ismir2019/paper/000033.pdf).
    """
    def __init__(self, dim_in, profile, device='cpu'):
        """
        Initialize a SoftmaxGroups tablature layer.

        Parameters
        ----------
        See TranscriptionModel class...
        """

        super().__init__(dim_in, profile, device)

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        # Set the tablature layer to Softmax groups
        self.tablature_layer = SoftmaxGroups(dim_in, num_groups, num_classes)

    def pre_proc(self, batch):
        """
        Perform common symbolic tablature transcription pre-processing,
        and add all data to the specified device.

        Parameters
        ----------
        batch : dict
          Dictionary containing ground-truth tablature (and potentially multipitch) for a group of tracks

        Returns
        ----------
        batch : dict
          Dictionary containing ground-truth multipitch as features
        """

        # Perform pre-processing steps of parent class
        batch = super().pre_proc(batch)

        # Make sure all data is on correct device
        batch = tools.dict_to_device(batch, self.device)

        return batch

    def post_proc(self, batch):
        """
        Calculate tablature loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature and potentially loss
        """

        # Calculate tablature loss
        output = super().post_proc(batch)

        # Finalize the tablature estimation
        output[tools.KEY_TABLATURE] = self.tablature_layer.finalize_output(output[tools.KEY_TABLATURE])

        return output


class LogisticTablatureEstimator(TablatureEstimator):
    """
    Multi-unit (string/fret) logistic regression tablature layer with pairwise inhibition.
    """
    def __init__(self, dim_in, profile, matrix_path=None, silence_activations=False, device='cpu'):
        """
        Initialize a LogisticBank tablature layer and the inhibition matrix.

        Parameters
        ----------
        See TranscriptionModel class for others...
        matrix_path : str or None (optional)
          Path to inhibition matrix
        silence_activations : bool
          Whether to explicitly model silence
        """

        super().__init__(dim_in, profile, device)

        # Extract tablature parameters
        num_strings = self.profile.get_num_dofs()
        num_pitches = self.profile.num_pitches

        # Calculate output dimensionality
        dim_out = num_strings * num_pitches

        self.silence_activations = silence_activations

        if self.silence_activations:
            # Account for no-string activations
            dim_out += num_strings

        # Set the tablature layer to a Logistic bank
        self.tablature_layer = LogisticBank(dim_in, dim_out)

        if matrix_path is None:
            # Default the inhibition matrix if it does not exist (inhibit string groups)
            inhibition_matrix = self.initialize_default_matrix(self.profile, self.silence_activations)
        else:
            # Load the inhibition matrix at the given path
            inhibition_matrix = load_inhibition_matrix(matrix_path)
            # Trim the inhibition matrix to match the chosen profile
            inhibition_matrix = torch.Tensor(trim_inhibition_matrix(inhibition_matrix, num_strings, num_pitches, self.silence_activations))

        # Initialize the inhibition matrix and add it to the specified device
        self.inhibition_matrix = inhibition_matrix.to(self.device)

    @staticmethod
    def initialize_default_matrix(profile, silence_activations):
        """
        Calculate the inhibition loss for frame-level logistic
        tablature predictions, given a pre-existing inhibition matrix.

        Parameters
        ----------
        profile : TablatureProfile (tools/instrument.py)
          Instructions for organizing tablature into logistic activations
        silence_activations : bool
          Whether the silent string is explicitly modeled as an activation

        Returns
        ----------
        inhibition_matrix : ndarray (N x N)
          Matrix of inhibitory weights for string/fret pairs
          N - number of unique string/fret activations
        """

        # Extract tablature parameters
        num_strings = profile.get_num_dofs()
        num_pitches = profile.num_pitches

        # Calculate output dimensionality
        dim_out = num_strings * num_pitches

        if silence_activations:
            # Account for no-string activations
            dim_out += num_strings

        # Create a identity matrix with size equal to number of strings
        inhibition_matrix = torch.eye(num_strings)
        # Repeat the matrix along both dimensions for each pitch
        inhibition_matrix = torch.repeat_interleave(inhibition_matrix, num_pitches + int(silence_activations), dim=0)
        inhibition_matrix = torch.repeat_interleave(inhibition_matrix, num_pitches + int(silence_activations), dim=1)
        # Subtract out self-connections
        inhibition_matrix = inhibition_matrix - torch.eye(dim_out)

        return inhibition_matrix

    def pre_proc(self, batch):
        """
        Perform common symbolic tablature transcription pre-processing, re-organize the
        ground-truth as logistic activations, and add all data to the specified device.

        Parameters
        ----------
        batch : dict
          Dictionary containing ground-truth tablature for a group of tracks

        Returns
        ----------
        batch : dict
          Dictionary containing logistic tablature ground-truth
        """

        # Perform pre-processing steps of parent class
        batch = super().pre_proc(batch)

        if tools.query_dict(batch, tools.KEY_TABLATURE):
            # Extract the tablature from the ground-truth
            tablature = batch[tools.KEY_TABLATURE]
            # Convert the tablature to logistic activations
            logistic = tools.tablature_to_logistic(tablature, self.profile, silence=self.silence_activations)
            # Add back to the ground-truth
            batch[tools.KEY_TABLATURE] = logistic

        # Make sure all data is on correct device
        batch = tools.dict_to_device(batch, self.device)

        return batch

    @staticmethod
    def calculate_inhibition_loss(logistic_tablature, inhibition_matrix):
        """
        Calculate the inhibition loss for frame-level logistic
        tablature predictions, given a pre-existing inhibition matrix.

        Parameters
        ----------
        logistic_tablature : tensor (T x N)
          Tensor of tablature activations (e.g. string/fret combinations)
          T - number of frames
          N - number of unique string/fret activations
        inhibition_matrix : ndarray (N x N)
          Matrix of inhibitory weights for string/fret pairs
          N - number of unique string/fret activations

        Returns
        ----------
        inhibition_loss : float
          Measure of the degree to which inhibitory pairs are co-activated
        """

        # Determine the number of unique activations
        num_activations = inhibition_matrix.shape[-1]

        # Compute the outer product of the string/fret activations
        outer = torch.bmm(logistic_tablature.view(-1, num_activations, 1),
                          logistic_tablature.view(-1, 1, num_activations))
        # Un-collapse the batch dimension
        outer = outer.view(tuple(logistic_tablature.shape[:-1]) + tuple([num_activations] * 2))
        # Apply the inhibition matrix weights to the outer product
        inhibition_loss = inhibition_matrix * outer

        # Average the inhibition loss over the batch and frame dimension
        inhibition_loss = torch.mean(torch.mean(inhibition_loss, axis=0), axis=0)

        # Divide by two, since every pair will have a duplicate entry, and sum across pairs
        inhibition_loss = torch.sum(inhibition_loss / 2)

        return inhibition_loss

    def post_proc(self, batch):
        """
        Calculate tablature/inhibition loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature and potentially loss
        """

        # Calculate tablature loss
        output = super().post_proc(batch)

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Unpack the loss if it exists
        loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

        if loss is None:
            # Create a new dictionary to hold the loss
            loss = {}

        # Apply the sigmoid activation here (since we are using BCE loss with logits)
        tablature_est = torch.sigmoid(tablature_est)

        # Determine if loss is being tracked
        if total_loss:
            # Compute the inhibition loss for the estimated tablature
            inhibition_loss = self.calculate_inhibition_loss(tablature_est, self.inhibition_matrix)
            # Add the inhibition loss to the tracked loss dictionary
            loss[tools.KEY_LOSS_INH] = inhibition_loss
            # Add the inhibition loss to the total loss
            # TODO - the following line can be used for annealing
            # total_loss = (total_loss * (50000 - self.iter) / 50000) + (inhibition_loss * (self.iter / 50000))
            total_loss += inhibition_loss

        # Determine if loss is being tracked
        if total_loss:
            # Add the loss to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        """
        # TODO - verify it still works with silence activations
        # 6th-order greedy choice algorithm
        num_strings = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + int(self.silence_activations)
        B, T, A = tablature_est.size()

        #correlation = (1 - self.weights).unsqueeze(0).unsqueeze(0).to(tablature_est.device)
        correlation = torch.tile((1 - self.inhibition_matrix), (B, T, 1, 1)).to(tablature_est.device)
        for i in range(num_strings):
            likelihood = torch.mul(tablature_est.unsqueeze(-2), correlation)
            #likelihood = likelihood.view(B, T, A, num_strings, num_classes)

            best_combo_score = torch.sum(likelihood, dim=-1)
            best_combo_idx = torch.argmax(best_combo_score, dim=-1)
            best_combo_idx = best_combo_idx.unsqueeze(-1).repeat((1, 1, A))
            best_combo_str = best_combo_idx // num_classes

            zero_idcs_str = num_classes * best_combo_str
            zero_idcs_stp = zero_idcs_str + num_classes

            tablature_idcs = torch.tile(torch.arange(A), (B, T, 1)).to(tablature_est.device)
            gt_idcs = tablature_idcs >= zero_idcs_str
            lt_idcs = tablature_idcs < zero_idcs_stp
            non_max_idx = torch.logical_not(tablature_idcs == best_combo_idx)

            zero_string_rows = torch.logical_and(gt_idcs, lt_idcs)
            zero_non_max_rows = torch.logical_and(zero_string_rows, non_max_idx)

            tablature_est[zero_non_max_rows] = 0
            correlation[zero_string_rows] = 0
        """

        """"""
        # Transpose the frame and string/fret combination dimension
        tablature_est = tablature_est.transpose(-2, -1)

        # Take the argmax per string for the final predictions
        tablature_est = tools.logistic_to_tablature(tablature_est, self.profile, self.silence_activations, silence_thr=0.25)
        """"""

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = tablature_est

        return output
