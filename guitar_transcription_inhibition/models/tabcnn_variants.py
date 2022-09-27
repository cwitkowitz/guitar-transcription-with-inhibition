# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import TabCNN, OnlineLanguageModel
from . import LogisticTablatureEstimator

import amt_tools.tools as tools

# Regular imports
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

        # Break off the output layer so it can be referenced
        dense_core, tablature_layer = self.dense[:-1], self.dense[-1]

        # Insert the recurrent layer before the output layer
        self.dense = torch.nn.Sequential(
            dense_core,
            OnlineLanguageModel(dim_in=tablature_layer.dim_in,
                                dim_out=tablature_layer.dim_in),
            tablature_layer
        )


class TabCNNLogistic(TabCNN):
    """
    Implements TabCNN with a logistic output layer instead of the classic (softmax) output layer.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, matrix_path=None,
                 silence_activations=False, lmbda=1, device='cpu'):
        """
        Initialize the model and replace the final layer.

        Parameters
        ----------
        See TabCNN class for others...
        matrix_path : str or None (optional)
          Path to inhibition matrix
        silence_activations : bool
          Whether to explicitly model silence
        lmbda : float
          Multiplier for the inhibition loss
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Break off the output layer to establish an explicit reference
        self.dense, self.tablature_layer = self.dense[:-1], self.dense[-1]

        # Replace the tablature layer with a logistic tablature estimator
        self.tablature_layer = LogisticTablatureEstimator(dim_in=self.tablature_layer.dim_in,
                                                          profile=profile,
                                                          matrix_path=matrix_path,
                                                          silence_activations=silence_activations,
                                                          lmbda=lmbda,
                                                          device=device)

    def change_device(self, device=None):
        """
        Change the device and load the model and output layer onto the new device.

        Parameters
        ----------
        device : string, int or None, optional (default None)
          Device to load model onto
        """

        super().change_device(device)

        # Update the tracked device of the tablature layer
        self.tablature_layer.change_device(device)

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
        batch = self.tablature_layer.pre_proc(batch)

        # Perform TabCNN pre-processing steps
        batch = super().pre_proc(batch)

        return batch

    def forward(self, feats):
        """
        Perform the main processing steps for TabCNN.

        Parameters
        ----------
        feats : Tensor (B x T x C x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          C - number of channels in features
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

        # Extract the embeddings from the output dictionary (labeled as tablature)
        embeddings = output.pop(tools.KEY_TABLATURE)

        # Process the embeddings with the output layer
        output = self.tablature_layer(embeddings)

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

        # Call the post-processing method of the tablature layer
        output = self.tablature_layer.post_proc(batch)

        return output


class TabCNNLogisticRecurrent(TabCNNLogistic):
    """
    Implements TabCNNLogistic with a recurrent layer inserted before output layer.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, matrix_path=None,
                 silence_activations=False, lmbda=1, device='cpu'):
        """
        Initialize the model and insert the recurrent layer.

        Parameters
        ----------
        See TabCNNLogistic class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity,
                         matrix_path, silence_activations, lmbda, device)

        # Insert a recurrent layer at the end of the dense core
        self.dense.append(OnlineLanguageModel(dim_in=self.tablature_layer.dim_in,
                                              dim_out=self.tablature_layer.dim_in))
