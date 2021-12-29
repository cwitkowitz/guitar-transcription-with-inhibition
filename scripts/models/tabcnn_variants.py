# My imports
from amt_tools.models import TabCNN, LanguageModel, LogisticBank

import amt_tools.tools as tools

from .tablature_layers import ClassicTablatureEstimator, LogisticTablatureEstimator

# Regular imports
from copy import deepcopy
import torch


class TabCNNRecurrent(TabCNN):
    """
    Implements TabCNN with a recurrent layer inserted before output layer.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and insert the recurrent layer.

        Parameters
        ----------
        See TabCNN class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Insert the recurrent layer before the output layer
        self.dense = torch.nn.Sequential(
            self.dense[:-1],
            LanguageModel(dim_in=128, dim_out=128, bidirectional=False),
            self.dense[-1]
        )


class TabCNNLogistic(TabCNNRecurrent):
    """
    Implements TabCNN with a logistic output layer instead of the classic (softmax) output layer.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1,
                 matrix_path=None, silence_activations=False, device='cpu'):
        """
        Initialize the model and replace the final layer.

        Parameters
        ----------
        See TabCNN class for others...
        matrix_path : str or None (optional)
          Path to inhibition matrix
        silence_activations : bool
          Whether to explicitly model silence
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Replace the tablature layer with a logistic tablature estimator
        self.dense[-1] = LogisticTablatureEstimator(128, profile, matrix_path, silence_activations, device)

    def pre_proc(self, batch):
        """
        Perform necessary pre-processing steps for the transcription model.

        Parameters
        ----------
        batch : dict
          Dictionary containing all relevant fields for a group of tracks

        Returns
        ----------
        batch : dict
          Dictionary with all PyTorch Tensors added to the appropriate device
          and all pre-processing steps complete
        """

        # Perform output layer pre-processing steps
        batch = self.dense[-1].pre_proc(batch)

        # Perform TabCNN pre-processing steps
        batch = super().pre_proc(batch)

        return batch

    def forward(self, feats):
        """
        Perform the main processing steps for TabCNN.

        Parameters
        ----------
        feats : Tensor (B x T x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Run the standard steps
        output = super().forward(feats)

        # Do not double-pack the tablature
        output = output.pop(tools.KEY_TABLATURE)

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
          Dictionary containing tablature as well as loss
        """

        # Obtain a pointer to the output layer
        tablature_output_layer = self.dense[-1]

        # Call the post-processing method of the tablature layer
        output = tablature_output_layer.post_proc(batch)

        return output


class TabCNNMultipitch(TabCNN):
    """
    Implements TabCNN for multipitch estimation instead of tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and replace the final layer.

        Parameters
        ----------
        See TabCNN class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Determine the number of input neurons to the current softmax layer
        n_neurons = self.dense[-1].dim_in

        # Determine the number of distinct pitches
        n_multipitch = self.profile.get_range_len()

        # Create a layer for multipitch estimation and replace the tablature layer
        self.dense[-1] = LogisticBank(n_neurons, n_multipitch)

    def forward(self, feats):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        feats : Tensor (B x T x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ multipitch Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Run the standard steps
        output = super().forward(feats)

        # Correct the label from tablature to multipitch
        output[tools.KEY_MULTIPITCH] = output.pop(tools.KEY_TABLATURE)

        return output

    def post_proc(self, batch):
        """
        Calculate multipitch loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multipitch and potentially loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain a pointer to the output layer
        multipitch_output_layer = self.dense[-1]

        # Obtain the multipitch estimation
        multipitch_est = output[tools.KEY_MULTIPITCH]

        # Keep track of loss
        total_loss = 0

        # Check to see if ground-truth multipitch is available
        if tools.KEY_MULTIPITCH in batch.keys():
            # Calculate the loss and add it to the total
            # TODO - add in with multipitch loss label?
            total_loss += multipitch_output_layer.get_loss(multipitch_est, batch[tools.KEY_MULTIPITCH])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize multipitch estimation
        output[tools.KEY_MULTIPITCH] = multipitch_output_layer.finalize_output(multipitch_est)

        return output


class TabCNNJointCustom(TabCNNMultipitch):
    """
    Implements TabCNN for joint (sequential) multipitch and custom tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, threshold=0.5, detach=True, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See TabCNNMultipitch class for others...
        threshold : float (0 <= threshold <= 1)
          Threshold for positive multipitch activations (0 to disable thresholding)
        detach : bool
          Whether to prevent tablature estimation gradient from propagating through multipitch estimation
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Fields for joint processing
        self.threshold = threshold
        self.detach = detach

        # Initialize a null tablature output layer
        self.tablature_layer = None

    def set_tablature_layer(self, tablature_layer):
        """
        Helper function to allow for overwriting the tablature layer.

        Note: if you wish to train the tablature layer, make sure it exists before calling self.parameters()
        # TODO - make sure this doesn't cause problems with gradient or loss computation

        Parameters
        ----------
        tablature_layer : TranscriptionModel or None
          Preexisting tablature estimation layer to use for joint processing
        """

        self.tablature_layer = tablature_layer

    def forward(self, feats):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        feats : Tensor (B x T x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ multipitch and tablature Tensors (B x T x O1/O2)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O1 - number of multipitch neurons
          O2 - number of tablature neurons
        """

        # Run the multipitch estimation steps
        output = super().forward(feats)

        if self.tablature_layer is not None:
            # Extract the multipitch activations from the output
            multipitch = output[tools.KEY_MULTIPITCH].clone()

            if self.threshold:
                # Threshold the multipitch activations
                multipitch = tools.threshold_activations(multipitch, self.threshold)

            if self.detach:
                # Do not propagate tablature gradient through multipitch estimation
                multipitch = multipitch.detach()

            # Determine how many frets are missing from the multipitch
            num_missing_frets = self.tablature_layer.profile.get_range_len() - self.profile.get_range_len()

            # Create the zero padding to add to the multipitch
            padding = torch.zeros((multipitch.shape[:-1] + tuple([num_missing_frets]))).to(multipitch.device)

            # Pad the multipitch to account for the missing frets
            multipitch = torch.cat((multipitch, padding), dim=-1)

            # Obtain the tablature estimate and add it to the output dictionary
            #output.update(self.tablature_layer(multipitch))
            output.update(self.tablature_layer.run_on_batch({tools.KEY_FEATS : multipitch}))

        return output

    def post_proc(self, batch):
        """
        Calculate multipitch/tablature loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multipitch/tablature and potentially loss
        """

        # Call the parent function to do the multipitch stuff
        batch[tools.KEY_OUTPUT] = super().post_proc(batch)

        #if self.tablature_layer is not None:
        #    # Perform the post-processing steps of the tablature layer
        #    output = self.tablature_layer.post_proc(batch)
        #else:
        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        return output


class TabCNNJoint(TabCNNJointCustom):
    """
    Implements TabCNN for joint (sequential) multipitch and tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, threshold=0.5, detach=True, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See TabCNNJointCustom...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, 0.5, True, device)

        # Fields for joint processing
        self.threshold = threshold
        self.detach = detach

        # Determine the number of distinct pitches
        n_multipitch = self.profile.get_range_len()

        # Initialize the tablature layer as Softmax groups
        tablature_layer = ClassicTablatureEstimator(n_multipitch, profile, device)

        # Set the tablature layer to the Softmax groups
        self.set_tablature_layer(tablature_layer)
