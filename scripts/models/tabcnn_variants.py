# My imports
from amt_tools.models import TabCNN, LogisticBank, SoftmaxGroups

import amt_tools.tools as tools

# Regular imports
from copy import deepcopy


class TabCNNMultipitch(TabCNN):
    """
    Implements TabCNN for multipitch estimation instead of tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

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
            total_loss += multipitch_output_layer.get_loss(multipitch_est, batch[tools.KEY_MULTIPITCH])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize multipitch estimation
        output[tools.KEY_MULTIPITCH] = multipitch_output_layer.finalize_output(multipitch_est)

        return output


class TabCNNJoint(TabCNNMultipitch):
    """
    Implements TabCNN for joint (sequential) multipitch and tablature estimation.
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
          Whether to disallow tablature estimation gradient to propagate through multipitch estimation
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Fields for joint processing
        self.threshold = threshold
        self.detach = detach

        # Determine the number of distinct pitches
        n_multipitch = self.profile.get_range_len()

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        # Initialize the tablature layer
        self.tablature_layer = SoftmaxGroups(n_multipitch, num_groups, num_classes)

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

        # Extract the multipitch activations from the output
        multipitch = output[tools.KEY_MULTIPITCH]

        if self.threshold:
            # Threshold the multipitch activations
            multipitch = tools.threshold_activations(multipitch, self.threshold)

        if self.detach:
            # Do not propagate tablature gradient through multipitch estimation
            multipitch = multipitch.detach()

        # Obtain the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.tablature_layer(multipitch)

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
        output = super().post_proc(batch)

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Keep track of loss
        total_loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if total_loss is None else total_loss[tools.KEY_LOSS_TOTAL]

        # Check to see if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Calculate the loss and add it to the total
            total_loss += self.tablature_layer.get_loss(tablature_est, batch[tools.KEY_TABLATURE])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = self.tablature_layer.finalize_output(tablature_est)

        return output
