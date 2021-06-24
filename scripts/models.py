# My imports
from amt_models.models import *

import amt_models.tools as tools

# Regular imports
from copy import deepcopy

import torch.nn as nn

# TODO - All variants for OF2


class OnsetsFramesTablature(OnsetsFrames):
    """
    Implements the Onsets & Frames model (V1), where all logistic
    banks are replaced with tablature-friendly softmax groups.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=2, detach_heads=False, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionModel class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, detach_heads, device)

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        # Exchange each head's output layer with softmax groups
        self.onset_head[-1] = SoftmaxGroups(self.dim_lm, num_groups, num_classes)
        self.pitch_head[-1] = SoftmaxGroups(self.dim_am, num_groups, num_classes)

        # Re-initialize the refinement layer with new input size and softmax groups
        self.dim_aj = 2 * num_groups * num_classes
        self.adjoin = nn.Sequential(
            LanguageModel(self.dim_aj, self.dim_lm),
            SoftmaxGroups(self.dim_lm, num_groups, num_classes)
        )

    def post_proc(self, batch):
        """
        Calculate loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature and onsets output as well as loss
        """

        # Create a local copy of the batch so it is only modified within scope
        # TODO
        # batch = deepcopy(batch)

        # Check to see if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Change the label so as to comply with super's post-processing
            batch[tools.KEY_MULTIPITCH] = batch.pop(tools.KEY_TABLATURE)

            # Add ground-truth onsets if they are not available
            if tools.KEY_ONSETS not in batch.keys():
                # Obtain the onset labels from the reference tablature
                stacked_multi_pitch = tools.tablature_to_stacked_multi_pitch(batch[tools.KEY_MULTIPITCH])
                stacked_onsets = tools.stacked_multi_pitch_to_stacked_onsets(stacked_multi_pitch)
                batch[tools.KEY_ONSETS] = tools.stacked_multi_pitch_to_tablature(stacked_onsets, self.profile)

        # Perform the standard post-processing steps
        output = super().post_proc(batch)

        # Correct the pitch label from multi pitch to tablature
        output[tools.KEY_TABLATURE] = output.pop(tools.KEY_MULTIPITCH)

        # TODO - change loss label to tablature

        return output


class OnsetsFramesSingle(OnsetsFramesTablature):
    """
    Implements the tablature-friendly Onsets & Frames model (V1), with uni-directional LSTMs.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=2, detach_heads=False, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionModel and OnsetsFrames class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, detach_heads, device)

        # Exchange bi-LSTMs with uni-LSTMs (same output size)
        self.onset_head[-2] = LanguageModel(self.dim_am, self.dim_lm, bidirectional=False)
        self.adjoin[-2] = LanguageModel(self.dim_aj, self.dim_lm, bidirectional=False)


class OnsetsFramesTwoStage(OnsetsFrames):
    """
    Implements a multi-stage Onsets & Frames model (V1), where the first pitch head
    predicts multi pitch activations and the second pitch head predicts tablature.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=2, detach_heads=False, device='cpu'):
        super().__init__(dim_in, profile, in_channels, model_complexity, detach_heads, device)

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        # Exchange the refiner head's output layer with softmax groups
        self.adjoin[-1] = SoftmaxGroups(self.dim_lm, num_groups, num_classes)

    def forward(self, feats):
        """
        Perform the standard processing steps for Onsets & Frames (V1),
        while differentiating between multi pitch and tablature activations.

        Parameters
        ----------
        feats : Tensor (B x C x T x F)
          Input features for a batch of tracks,
          B - batch size
          C - channels
          T - number of frames
          F - number of features (frequency bins)

        Returns
        ----------
        output : dict w/ Tensors (B x T x O)
          Dictionary containing multi pitch and onsets output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out) (different for multi pitch vs tablature layers)
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the initial multi pitch estimate
        multi_pitch = self.pitch_head(feats)
        output[tools.KEY_MULTIPITCH] = multi_pitch

        # Obtain the onsets estimate and add it to the output dictionary
        onsets = self.onset_head(feats)
        output[tools.KEY_ONSETS] = onsets

        # Concatenate the above estimates
        joint = torch.cat((onsets, multi_pitch), -1)

        # Obtain a refined multi pitch estimate in tablature
        # format and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.adjoin(joint)

        return output

    def post_proc(self, batch):
        """
        Calculate loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature, multi pitch and onsets output as well as loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain pointers to the output layers
        onset_output_layer = self.onset_head[-1]
        pitch_output_layer = self.pitch_head[-1]
        tabs_output_layer = self.adjoin[-1]

        # Obtain the onset, pitch, and tablature estimations
        onsets_est = output[tools.KEY_ONSETS]
        multi_pitch_est = output[tools.KEY_MULTIPITCH]
        tablature_est = output[tools.KEY_TABLATURE]

        # Check if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Keep track of all losses
            loss = dict()

            # Calculate the tablature loss term
            tablature_ref = batch[tools.KEY_TABLATURE]
            tablature_loss = tabs_output_layer.get_loss(tablature_est, tablature_ref)
            loss[tools.KEY_LOSS_TABS] = tablature_loss

            # Check to see if ground-truth multi pitch is available
            if tools.KEY_MULTIPITCH in batch.keys():
                # Extract the ground-truth and calculate the multi pitch loss term
                multi_pitch_ref = batch[tools.KEY_MULTIPITCH]
            else:
                # Obtain the multi pitch labels from the reference tablature
                multi_pitch_ref = tools.tablature_to_stacked_multi_pitch(tablature_ref, self.profile)
                multi_pitch_ref = tools.stacked_multi_pitch_to_multi_pitch(multi_pitch_ref)

            # Calculate the multi pitch loss term
            multi_pitch_loss = pitch_output_layer.get_loss(multi_pitch_est, multi_pitch_ref)
            loss[tools.KEY_LOSS_PITCH] = multi_pitch_loss

            # Check to see if ground-truth onsets are available
            if tools.KEY_ONSETS in batch.keys():
                # Extract the ground-truth
                onsets_ref = batch[tools.KEY_ONSETS]
            else:
                # Obtain the onset labels from the reference multi pitch
                onsets_ref = tools.multi_pitch_to_onsets(multi_pitch_ref)

            # Calculate the onsets loss term
            onsets_loss = onset_output_layer.get_loss(onsets_est, onsets_ref)
            loss[tools.KEY_LOSS_ONSETS] = onsets_loss

            # Compute the total loss and add it to the output dictionary
            output[tools.KEY_LOSS] = tablature_loss + multi_pitch_loss + onsets_loss
            total_loss = tablature_loss + onsets_loss
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        # Finalize onset and pitch estimations
        output[tools.KEY_ONSETS] = onset_output_layer.finalize_output(onsets_est)
        output[tools.KEY_MULTIPITCH] = pitch_output_layer.finalize_output(multi_pitch_est)
        output[tools.KEY_TABLATURE] = tabs_output_layer.finalize_output(tablature_est)

        return output


class CausalBasicLogistic(TranscriptionModel):
    def __init__(self, dim_in, profile, in_channels=1, model_complexity=1, device='cpu'):

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Number of frames required for a prediction
        self.frame_width = 9

        # Number of output neurons for each head's activations
        dim_out = self.profile.get_range_len()

        # Number of output neurons for the acoustic models
        self.dim_am = 256 * self.model_complexity

        # Create the pitch detector head
        self.pitch_head = AcousticModel(self.dim_in, self.dim_am, self.in_channels, self.model_complexity)

        # Determine the total size of the feature map
        self.feat_map_size = self.frame_width * self.dim_am

        # Create the output layer
        self.output_layer = LogisticBank(self.feat_map_size, dim_out)

    def pre_proc(self, batch):
        batch = super().pre_proc(batch)

        # Create a local copy of the batch so it is only modified within scope
        # TODO
        # batch = deepcopy(batch)

        # Extract the features from the batch as a NumPy array
        feats = tools.tensor_to_array(batch[tools.KEY_FEATS])
        # Window the features to mimic real-time operation
        feats = tools.framify_activations(feats, self.frame_width)
        # Convert the features back to PyTorch tensor and add to device
        feats = tools.array_to_tensor(feats, self.device)
        # Switch the sequence-frame and feature axes
        feats = feats.transpose(-2, -3)
        # Remove the single channel dimension
        feats = feats.squeeze(1)

        # Switch the frequency and time axes
        feats = feats.transpose(-1, -2)

        batch[tools.KEY_FEATS] = feats

        return batch

    def forward(self, feats):
        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the batch size before sequence-frame axis is collapsed
        batch_size = feats.size(0)

        # Collapse the sequence-frame axis into the batch axis,
        # so that each windowed group of frames is treated as one
        # independent sample. This is not done during pre-processing
        # in order to maintain consistency with the notion of batch size
        feats = feats.reshape(-1, 1, self.frame_width, self.dim_in)

        # Obtain the feature embeddings
        embeddings = self.pitch_head(feats)
        # Flatten spatial features into one embedding
        embeddings = embeddings.flatten(1)
        # Size of the embedding
        embedding_size = embeddings.size(-1)
        # Restore proper batch dimension, unsqueezing sequence-frame axis
        embeddings = embeddings.view(batch_size, -1, embedding_size)

        # Obtain the multi pitch estimate and add it to the output dictionary
        output[tools.KEY_MULTIPITCH] = self.output_layer(embeddings)

        return output

    def post_proc(self, batch):

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain the multi pitch estimation
        multi_pitch_est = output[tools.KEY_MULTIPITCH]

        # Check to see if ground-truth multi pitch is available
        if tools.KEY_MULTIPITCH in batch.keys():
            # Extract the ground-truth, calculate the loss and add it to the dictionary
            multi_pitch_ref = batch[tools.KEY_MULTIPITCH]
            multi_pitch_loss = self.output_layer.get_loss(multi_pitch_est, multi_pitch_ref)
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : multi_pitch_loss}

        # Finalize multi pitch estimation
        output[tools.KEY_MULTIPITCH] = self.output_layer.finalize_output(multi_pitch_est)

        return output


class CausalBasicSoftmax(CausalBasicLogistic):
    def __init__(self, dim_in, profile, in_channels=1, model_complexity=1, device='cpu'):

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        # Exchange the output layer with softmax groups
        self.output_layer = SoftmaxGroups(self.feat_map_size, num_groups, num_classes)

    def post_proc(self, batch):
        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain the tablature estimation (has multi pitch label from parent class)
        tablature_est = output[tools.KEY_MULTIPITCH]

        # Check to see if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Extract the ground-truth, calculate the loss and add it to the dictionary
            tablature_ref = batch[tools.KEY_TABLATURE]
            tablature_loss = self.output_layer.get_loss(tablature_est, tablature_ref)
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : tablature_loss}

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = self.output_layer.finalize_output(tablature_est)

        return output