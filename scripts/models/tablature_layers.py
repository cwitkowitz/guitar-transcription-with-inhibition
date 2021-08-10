# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import TranscriptionModel, SoftmaxGroups, LanguageModel, LogisticBank
import amt_tools.tools as tools

# Regular imports
from math import ceil

import numpy as np
import torch


class ClassicTablatureEstimator(TranscriptionModel):
    """
    A model wrapper for the Softmax groups output layer from
    TabCNN (http://archives.ismir.net/ismir2019/paper/000033.pdf).
    """
    def __init__(self, dim_in, profile, device='cpu'):
        """
        Initialize the tablature layer and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionModel class...
        """

        super().__init__(dim_in, profile, 1, 1, 1, device)

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        # Initialize the tablature layer as Softmax groups
        self.tablature_layer = SoftmaxGroups(dim_in, num_groups, num_classes)

    def pre_proc(self, batch):
        """
        Perform necessary pre-processing steps for the tablature layer.

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
        elif tools.query_dict(batch, tools.KEY_FEATS):
            # Extract the multipitch from the ground-truth
            multipitch = batch[tools.KEY_FEATS]
        else:
            # This will cause an error
            multipitch = None

        # Flip the dimensions of the multipitch and add as features to the dictionary
        batch[tools.KEY_FEATS] = multipitch.transpose(-1, -2)

        # Make sure all data is on correct device
        batch = tools.dict_to_device(batch, self.device)

        return batch

    def forward(self, multipitch):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        multipitch : Tensor (B x T x F)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (unique pitches)

        Returns
        ----------
        output : dict w/ tablature Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames
          O - number of tablature neurons
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Compute the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.tablature_layer(multipitch.float())

        return output

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
          Dictionary containing ablature and potentially loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Keep track of loss
        total_loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if total_loss is None else total_loss[tools.KEY_LOSS_TOTAL]

        # Check to see if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Calculate the loss and add it to the total
            # TODO - add in with tab loss label?
            total_loss += self.tablature_layer.get_loss(tablature_est, batch[tools.KEY_TABLATURE])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = self.tablature_layer.finalize_output(tablature_est)

        return output


class LogisticTablatureEstimator(ClassicTablatureEstimator):
    """
    TODO
    """
    def __init__(self, dim_in, profile, weights=None, device='cpu'):
        """
        TODO
        """

        super(ClassicTablatureEstimator, self).__init__(dim_in, profile, 1, 1, 1, device)

        # Extract tablature parameters
        num_strings = self.profile.get_num_dofs()
        num_pitches = self.profile.num_pitches

        # Calculate output dimensionality
        dim_out = num_strings * num_pitches

        # Initialize the tablature layer as a Logistic Bank
        self.tablature_layer = LogisticBank(dim_in, dim_out)

        weights = torch.Tensor(np.load('/home/rockstar/Desktop/guitar-transcription/generated/inhibition_matrix.npz')['inh'])

        # Default the weights connect string groups
        if weights is None:
            # Create a identity matrix with size equal to number of strings
            weights = torch.eye(num_strings)
            # Repeat the matrix along both dimensions for each pitch
            weights = torch.repeat_interleave(weights, num_pitches, dim=0)
            weights = torch.repeat_interleave(weights, num_pitches, dim=1)
            # Subtract out self-connections
            weights = weights - torch.eye(dim_out)

            """
            midi_tuning = self.profile.get_midi_tuning()

            for i in range(len(midi_tuning)):
                # Determine the offset until repeats for every other string
                offsets = midi_tuning - midi_tuning[i]
                # Bound the offsets by zero
                offsets[offsets < 0] = 0
                # Bound the offsets by the total number of pitches
                offsets[offsets >= self.profile.num_pitches] = 0

                for j in range(len(offsets)):
                    if not offsets[j]:
                        continue

                    row_start = i * self.profile.num_pitches + offsets[j]
                    row_stop = i * self.profile.num_pitches + self.profile.num_pitches

                    col_start = j * self.profile.num_pitches
                    col_stop = col_start + (row_stop - row_start)

                    row_range = torch.arange(row_start, row_stop)
                    col_range = torch.arange(col_start, col_stop)

                    weights[torch.cat((row_range, col_range)), torch.cat((col_range, row_range))] = 1
            """

        self.weights = weights.to(self.device)

    def pre_proc(self, batch):
        """
        TODO
        """

        if tools.query_dict(batch, tools.KEY_TABLATURE):
            # Extract the tablature from the ground-truth
            tablature = batch[tools.KEY_TABLATURE]
            # Convert to a stacked multi pitch array
            stacked_multi_pitch = tools.tablature_to_stacked_multi_pitch(tablature, self.profile)
            # Convert to logistic activations
            logistic = tools.stacked_multi_pitch_to_logistic(stacked_multi_pitch, self.profile)
            # Add back to the ground-truth
            batch[tools.KEY_TABLATURE] = logistic

        # Perform pre-processing steps of parent class
        batch = super().pre_proc(batch)

        return batch

    def forward(self, multipitch):
        """
        TODO
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Compute the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = torch.sigmoid(self.tablature_layer(multipitch.float()))

        return output

    def post_proc(self, batch):
        """
        TODO
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Keep track of loss
        total_loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if total_loss is None else total_loss[tools.KEY_LOSS_TOTAL]

        # Keep track of all losses
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
            # Determine the number of activations
            num_activations = self.weights.shape[-1]

            """
            # Loop through each string fret combo
            for i in range(num_activations // 2):
                for j in range(num_activations // 2):
                    # Add the inhibition penalty
                    inhibition_loss += torch.mean(self.weights[i, j] * tablature_est[..., i] * tablature_est[..., j])
            """

            # Compute the outer product of the string/fret activations
            outer = torch.bmm(tablature_est.view(-1, num_activations, 1),
                              tablature_est.view(-1, 1, num_activations))
            # Un-collapse the batch dimension
            outer = outer.view(tuple(tablature_est.shape[:-1]) + tuple([num_activations] * 2))
            # Apply the graph weights to the outer product
            inhibition_loss = self.weights * outer

            # Average the inhibition loss over the batch and frame dimension
            inhibition_loss = torch.mean(torch.mean(inhibition_loss, axis=0), axis=0)

            # Divide by two, since every pair will have a duplicate inhibition, and average for each pair
            inhibition_loss = torch.mean(inhibition_loss / 2)
            # Add the graph loss to the tracked loss dictionary
            loss[tools.KEY_LOSS_INH] = inhibition_loss
            # Add the inhibition loss to the total loss
            total_loss += inhibition_loss

        # Check to see if ground-truth multipitch is available
        #if tools.KEY_MULTIPITCH in batch.keys():
            # View the tablature activations as multipitch data
            #multipitch_est = tools.logistic_to_multi_pitch(tablature_est, self.profile)

        if total_loss:
            # Add the loss to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        # Obtain the final tablature estimate
        """
        # Threshold then softmax
        tablature_est = self.tablature_layer.finalize_output(tablature_est)
        tablature_est = tools.logistic_to_tablature(tablature_est, self.profile)
        """

        """"""
        # Choose maximum likelihood row
        B, T, A = tablature_est.size()
        likelihood = torch.mul(tablature_est.unsqueeze(-2), (1 - self.weights).unsqueeze(0).unsqueeze(0))
        likelihood = likelihood.view(B, T, A, self.profile.get_num_dofs(), self.profile.num_pitches)
        max_fret_vals, max_fret_idcs = torch.max(likelihood, dim=-1)

        threshold = 5E-4

        max_fret_vals[max_fret_vals < threshold] = 0
        max_fret_idcs[max_fret_vals < threshold] = -1

        best_combo_score = torch.sum(max_fret_vals, dim=-1)
        best_combo_idx = torch.argmax(best_combo_score, dim=-1)

        idcs = torch.meshgrid(torch.arange(B), torch.arange(T)) + tuple([best_combo_idx.view(B, T).detach().cpu()])
        tablature_est = max_fret_idcs[idcs].transpose(-1, -2)
        """"""

        """
        # Ridge regression argmax_b {G = a^T b + b^T W b}
        tablature_est = -0.5 * torch.matmul(tablature_est, torch.inverse(1 - self.weights))
        """

        """
        # Ridge regression with constraint that activations must sum to 6
        W = 1 - self.weights
        W_i = torch.inverse(W)

        lmbda = (torch.sum(torch.matmul(-tablature_est, W_i), dim=-1, keepdim=True) - 12) / torch.sum(W_i)
        tablature_est = -0.5 * torch.matmul(tablature_est + lmbda, W_i)
        """

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = tablature_est

        return output


class ConvTablatureEstimator(LogisticTablatureEstimator):
    """
    Basic tablature layer with 1D convolution before the Softmax Groups.
    """
    def __init__(self, dim_in, profile, model_complexity=1, device='cpu'):
        """
        Initialize the tablature layer and establish parameter defaults in function signature.

        Parameters
        ----------
        See ClassicTablatureEstimator class for others...
        model_complexity : int, optional (default 1)
          Scaling parameter for size of model's components
        dropout : float
          Dropout rate post-convolution (0 to disable)
        """

        # Kernel size should be long enough to span most intervals
        kernel_size = 13

        # Kernel size for max pooling
        max_size = 2

        # Dropout rate for convolutional layers
        dropout = 0.

        # Scale the number of channels by the model complexity
        num_channels = 10 * model_complexity

        # Calculate the embedding size of the output of the convolutional layer
        embedding_size = num_channels * ceil(dim_in / max_size)

        # Define the input dimensionality of the SoftmaxGroups
        smax_dim_in = embedding_size // model_complexity

        # Call super to initialize the Softmax groups
        super().__init__(smax_dim_in, profile, None, device)

        # Keep track of parameters
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.embedding_size = embedding_size

        # Determine the padding amount on both sides
        self.padding = (self.kernel_size // 2, self.kernel_size // 2 - (1 - self.kernel_size % 2))

        self.layer1 = torch.nn.Sequential(
            # 1st convolution
            torch.nn.Conv1d(1, self.num_channels, self.kernel_size),
            # 1st batch normalization
            torch.nn.BatchNorm1d(self.num_channels),
            # Activation function
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            # 2nd convolution
            torch.nn.Conv1d(self.num_channels, self.num_channels, self.kernel_size),
            # 2nd batch normalization
            torch.nn.BatchNorm1d(self.num_channels),
            # Pad for the extra activation
            torch.nn.ConstantPad1d((0, dim_in % 2), 0),
            # Activation function
            torch.nn.ReLU(),
            # 1st reduction
            torch.nn.MaxPool1d(max_size),
            # 1st dropout
            torch.nn.Dropout(dropout)
        )

    def forward(self, multipitch):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        multipitch : Tensor (B x T x F)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (unique pitches)

        Returns
        ----------
        output : dict w/ tablature Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames
          O - number of tablature neurons
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the sizes of each dimension
        B, T, F = multipitch.size()

        # Collapse the frame dimension into the time dimension,
        # add a channel dimension, and covert to float32
        multipitch = multipitch.reshape(-1, 1, F).float()

        # Pad the multipitch so that convolution produces the same number of features per channel
        multipitch = torch.nn.functional.pad(multipitch, self.padding)

        # Run the multipitch through the convolutional layers to obtain feature embeddings
        embeddings = self.layer1(multipitch)
        embeddings = self.layer2(torch.nn.functional.pad(embeddings, self.padding))
        embeddings = embeddings.reshape(B, T, -1)

        # Compute the tablature estimate and add it to the output dictionary
        # TODO - call super function with embeddings as input
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings)

        return output


class RecConvTablatureEstimator(ConvTablatureEstimator):
    """
    Tablature layer with 1D convolution and an LSTM before the Softmax Groups.
    """

    def __init__(self, dim_in, profile, model_complexity=1, device='cpu'):
        """
        Initialize the tablature layer and establish parameter defaults in function signature.

        Parameters
        ----------
        See RecConvTablatureEstimator class for others...
        """

        # Call super to initialize the 1D Conv layer and Softmax groups
        super().__init__(dim_in, profile, model_complexity, device)

        # Define the input and output dimensionality of the LSTM
        lstm_dim_in = self.embedding_size
        lstm_dim_out = self.embedding_size // model_complexity

        # Instantiate the uni-directional LSTM to process the embeddings
        self.lstm = LanguageModel(lstm_dim_in, lstm_dim_out, bidirectional=False)

    def forward(self, multipitch):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        multipitch : Tensor (B x T x F)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (unique pitches)

        Returns
        ----------
        output : dict w/ tablature Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames
          O - number of tablature neurons
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the sizes of each dimension
        B, T, F = multipitch.size()

        # Collapse the frame dimension into the time dimension,
        # add a channel dimension, and covert to float32
        multipitch = multipitch.reshape(-1, 1, F).float()

        # Pad the multipitch so that convolution produces the same number of features per channel
        multipitch = torch.nn.functional.pad(multipitch, self.padding)

        # Run the multipitch through the convolutional layers to obtain feature embeddings
        embeddings = self.layer1(multipitch)
        embeddings = self.layer2(torch.nn.functional.pad(embeddings, self.padding))
        embeddings = embeddings.reshape(B, T, -1)

        # Run the embeddings through the LSTM
        embeddings = self.lstm(embeddings)

        # Compute the tablature estimate and add it to the output dictionary
        # TODO - call super function with embeddings as input
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings)

        return output
