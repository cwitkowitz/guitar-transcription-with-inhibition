# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .tablature_layers import LogisticTablatureEstimator
from amt_tools.models import TabCNN, LanguageModel

import amt_tools.tools as tools

# Regular imports
import torch.nn as nn
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
            OnlineLanguageModel(dim_in=128, dim_out=128),
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

        # Replace the tablature layer with a logistic datasets estimator
        self.dense[-1] = LogisticTablatureEstimator(128, profile, matrix_path, silence_activations, device)

    def change_device(self, device=None):
        """
        Change the device and load the model onto the new device.

        Parameters
        ----------
        device : string, int or None, optional (default None)
          Device to load model onto
        """

        super().change_device(device)

        # Update the tracked device of the tablature layer
        self.dense[-1].change_device(device)

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

        # Do not double-pack the datasets
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


class OnlineLanguageModel(LanguageModel):
    """
    Implements a uni-directional and online-capable LSTM language model.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize the language model and establish parameter defaults in function signature.

        Parameters
        ----------
        See LanguageModel class...
        """

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        # Determine the number of neurons
        self.hidden_size = self.dim_out

        # Initialize the LSTM
        self.mlm = nn.LSTM(input_size=self.dim_in,
                           hidden_size=self.hidden_size,
                           batch_first=True,
                           bidirectional=False)

        # Keep track of the hidden and cell state
        self.hidden = None
        self.cell = None

    def forward(self, in_feats):
        """
        Feed features through the music language model.

        Parameters
        ----------
        in_feats : Tensor (B x 1 x E)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          E - dimensionality of input embeddings (dim_in)

        Returns
        ----------
        out_feats : Tensor (B x 1 x E)
          Embeddings for a batch of tracks,
          B - batch size
          T - number of frames
          E - dimensionality of output embeddings (dim_out)
        """

        if self.training:
            # Call the regular forward function
            out_feats = super().forward(in_feats)
        else:
            # Process the chunk, using the previous hidden and cell state
            out_feats, (self.hidden, self.cell) = self.mlm(in_feats, (self.hidden, self.cell))

        return out_feats
