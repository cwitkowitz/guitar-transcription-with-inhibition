# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import TranscriptionModel, SoftmaxGroups
import amt_tools.tools as tools

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

        # Obtain the tablature from the ground-truth and make sure it consists of integers
        tablature = tools.unpack_dict(batch, tools.KEY_TABLATURE).to(torch.int32)

        # Convert the tablature to stacked multipitch arrays
        stacked_multipitch = tools.tablature_to_stacked_multi_pitch(tablature, self.profile)
        # Collapse into a single multipitch array
        multipitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multipitch)

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
            total_loss += self.tablature_layer.get_loss(tablature_est, batch[tools.KEY_TABLATURE])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = self.tablature_layer.finalize_output(tablature_est)

        return output
