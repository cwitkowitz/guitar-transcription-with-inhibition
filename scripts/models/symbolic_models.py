# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import LanguageModel

from .tablature_layers import ClassicTablatureEstimator, LogisticTablatureEstimator

# Regular imports
from math import ceil

import torch

# Note - the following two classes are essentially duplicates,
#        with the main difference being the parent tablature layer,
#        which controls the overhead functions that are executed.


class RecConvLogisticEstimator(LogisticTablatureEstimator):
    """
    Simple CRNN architecture to be used with the logistic output layer.
    """
    def __init__(self, profile, matrix_path=None, model_complexity=1, device='cpu'):
        """
        Initialize the components of the symbolic tablature model.

        Parameters
        ----------
        See LogisticTablatureEstimator class for others...
        model_complexity : int, optional (default 1)
          Scaling parameter for size of model's components
        """

        # Scale the number of channels by the model complexity
        num_channels = 10 * model_complexity

        # Determine the dimensionality of the multipitch "features"
        symbolic_dim_in = profile.get_range_len()

        # Kernel size for max pooling
        max_size = 2

        # Calculate the embedding size (output of the convolutional layer)
        embedding_size = num_channels * ceil(symbolic_dim_in / max_size)

        # Define the input dimensionality of the output layer
        output_layer_dim_in = embedding_size // model_complexity

        # Call the LogisticTablatureEstimator constructor
        super().__init__(dim_in=output_layer_dim_in, profile=profile, matrix_path=matrix_path, device=device)

        # Define the 1D convolutional kernel size (should be long enough to span most intervals)
        kernel_size = 13

        # Determine the padding amount on both sides
        self.padding = (kernel_size // 2, kernel_size // 2 - (1 - kernel_size % 2))

        # First 1D convolutional layer
        self.layer1 = torch.nn.Sequential(
            # 1st convolution
            torch.nn.Conv1d(1, num_channels, kernel_size),
            # 1st batch normalization
            torch.nn.BatchNorm1d(num_channels),
            # Activation function
            torch.nn.ReLU()
        )

        # Define the dropout rate for the second convolutional layer
        dropout = 0.

        # Second 1D convolutional layer
        self.layer2 = torch.nn.Sequential(
            # 2nd convolution
            torch.nn.Conv1d(num_channels, num_channels, kernel_size),
            # 2nd batch normalization
            torch.nn.BatchNorm1d(num_channels),
            # Pad for the extra activation
            torch.nn.ConstantPad1d((0, symbolic_dim_in % 2), 0),
            # Activation function
            torch.nn.ReLU(),
            # 1st reduction
            torch.nn.MaxPool1d(max_size),
            # 1st dropout
            torch.nn.Dropout(dropout)
        )

        # Define the output dimensionality of the LSTM
        lstm_dim_out = embedding_size // model_complexity

        # Instantiate the uni-directional LSTM to process the embeddings
        self.lstm = LanguageModel(embedding_size, lstm_dim_out, bidirectional=False)

    def forward(self, multipitch):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        multipitch : Tensor (B x T x F)
          Input multipitch for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of pitches

        Returns
        ----------
        output : dict w/ tablature Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames)
          O - tablature output dimensionality
        """

        # Obtain the sizes of each dimension
        B, T, F = multipitch.size()

        # Collapse the frame dimension into the time dimension, add a channel dimension, and covert to float32
        multipitch = multipitch.reshape(-1, 1, F).float()

        # Pad the multipitch so that convolution produces the same number of features per channel
        multipitch = torch.nn.functional.pad(multipitch, self.padding)

        # Run the multipitch through the first convolutional layer
        embeddings = self.layer1(multipitch)

        # Pad the embeddings a second time
        embeddings = torch.nn.functional.pad(embeddings, self.padding)

        # Run the multipitch through the second convolutional layer
        embeddings = self.layer2(embeddings)

        # Un-collapse the frame dimension
        embeddings = embeddings.reshape(B, T, -1)

        # Run the embeddings through the LSTM
        embeddings = self.lstm(embeddings)

        # Run the embeddings through the output layer
        output = super().forward(embeddings)

        return output


class RecConvClassicEstimator(ClassicTablatureEstimator):
    """
    Simple CRNN architecture to be used with the classic (softmax) output layer.
    """
    def __init__(self, profile, model_complexity=1, device='cpu'):

        # Scale the number of channels by the model complexity
        num_channels = 10 * model_complexity

        # Determine the dimensionality of the multipitch "features"
        symbolic_dim_in = profile.get_range_len()

        # Kernel size for max pooling
        max_size = 2

        # Calculate the embedding size (output of the convolutional layer)
        embedding_size = num_channels * ceil(symbolic_dim_in / max_size)

        # Define the input dimensionality of the output layer
        output_layer_dim_in = embedding_size // model_complexity

        # Call the ClassicTablatureEstimator constructor
        super().__init__(dim_in=output_layer_dim_in, profile=profile, device=device)

        # Define the 1D convolutional kernel size (should be long enough to span most intervals)
        kernel_size = 13

        # Determine the padding amount on both sides
        self.padding = (kernel_size // 2, kernel_size // 2 - (1 - kernel_size % 2))

        # First 1D convolutional layer
        self.layer1 = torch.nn.Sequential(
            # 1st convolution
            torch.nn.Conv1d(1, num_channels, kernel_size),
            # 1st batch normalization
            torch.nn.BatchNorm1d(num_channels),
            # Activation function
            torch.nn.ReLU()
        )

        # Define the dropout rate for the second convolutional layer
        dropout = 0.

        # Second 1D convolutional layer
        self.layer2 = torch.nn.Sequential(
            # 2nd convolution
            torch.nn.Conv1d(num_channels, num_channels, kernel_size),
            # 2nd batch normalization
            torch.nn.BatchNorm1d(num_channels),
            # Pad for the extra activation
            torch.nn.ConstantPad1d((0, symbolic_dim_in % 2), 0),
            # Activation function
            torch.nn.ReLU(),
            # 1st reduction
            torch.nn.MaxPool1d(max_size),
            # 1st dropout
            torch.nn.Dropout(dropout)
        )

        # Define the output dimensionality of the LSTM
        lstm_dim_out = embedding_size // model_complexity

        # Instantiate the uni-directional LSTM to process the embeddings
        self.lstm = LanguageModel(embedding_size, lstm_dim_out, bidirectional=False)

    def forward(self, multipitch):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        multipitch : Tensor (B x T x F)
          Input multipitch for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of pitches

        Returns
        ----------
        output : dict w/ tablature Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames)
          O - tablature output dimensionality
        """

        # Obtain the sizes of each dimension
        B, T, F = multipitch.size()

        # Collapse the frame dimension into the time dimension, add a channel dimension, and covert to float32
        multipitch = multipitch.reshape(-1, 1, F).float()

        # Pad the multipitch so that convolution produces the same number of features per channel
        multipitch = torch.nn.functional.pad(multipitch, self.padding)

        # Run the multipitch through the first convolutional layer
        embeddings = self.layer1(multipitch)

        # Pad the embeddings a second time
        embeddings = torch.nn.functional.pad(embeddings, self.padding)

        # Run the multipitch through the second convolutional layer
        embeddings = self.layer2(embeddings)

        # Un-collapse the frame dimension
        embeddings = embeddings.reshape(B, T, -1)

        # Run the embeddings through the LSTM
        embeddings = self.lstm(embeddings)

        # Run the embeddings through the output layer
        output = super().forward(embeddings)

        return output
