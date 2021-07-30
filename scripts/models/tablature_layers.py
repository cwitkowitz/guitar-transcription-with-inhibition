# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import TranscriptionModel, SoftmaxGroups, LanguageModel
import amt_tools.tools as tools

from .softmax_inhibition import InhibitedSoftmaxGroups

# Regular imports
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
        #pitch_ranges = torch.from_numpy(profile.get_dof_midi_range())
        #silent_class = torch.zeros((num_groups, 1))
        #membership = torch.cat((pitch_ranges, silent_class), dim=-1)
        #self.tablature_layer = InhibitedSoftmaxGroups(dim_in, num_groups, num_classes, membership)

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


class ConvTablatureEstimator(ClassicTablatureEstimator):
    """
    Basic tablature layer with 1D convolution before the Softmax Groups.
    """
    def __init__(self, dim_in, profile, model_complexity=1, dropout=0., device='cpu'):
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

        # Scale the number of channels by the model complexity
        num_channels = 10 * model_complexity

        # Calculate the embedding size of the output of the convolutional layer
        embedding_size = num_channels * dim_in

        # Define the input dimensionality of the SoftmaxGroups
        smax_dim_in = embedding_size // model_complexity

        # Call super to initialize the Softmax groups
        super().__init__(smax_dim_in, profile, device)

        # Keep track of parameters
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.embedding_size = embedding_size

        # Determine the padding amount on both sides
        self.padding = (self.kernel_size // 2, self.kernel_size // 2 - (1 - self.kernel_size % 2))

        # Instantiate a 1D convolutional layer
        self.conv_layer = torch.nn.Sequential(
            # Convolutional layer
            torch.nn.Conv1d(1, self.num_channels, self.kernel_size),
            # Activation function
            torch.nn.ReLU()
        )

        # Instantiate the dropout operation
        self.dropout = torch.nn.Dropout(dropout)

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

        # Run the multipitch through the convolutional layer to obtain feature embeddings
        embeddings = self.conv_layer(multipitch).reshape(B, T, -1)

        # Perform dropout regularization
        embeddings = self.dropout(embeddings)

        # Compute the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings)

        return output


class RecConvTablatureEstimator(ConvTablatureEstimator):
    """
    Tablature layer with 1D convolution and an LSTM before the Softmax Groups.
    """

    def __init__(self, dim_in, profile, model_complexity=1, dropout=0., device='cpu'):
        """
        Initialize the tablature layer and establish parameter defaults in function signature.

        Parameters
        ----------
        See RecConvTablatureEstimator class for others...
        """

        # Call super to initialize the 1D Conv layer and Softmax groups
        super().__init__(dim_in, profile, model_complexity, dropout, device)

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

        # Run the multipitch through the convolutional layer to obtain feature embeddings
        embeddings = self.conv_layer(multipitch).reshape(B, T, -1)

        # Perform dropout regularization
        embeddings = self.dropout(embeddings)

        # Run the embeddings through the LSTM
        embeddings = self.lstm(embeddings)

        # Compute the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings)

        return output
