# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import SoftmaxGroups
import amt_tools.tools as tools

# Regular imports
import torch


class InhibitedSoftmaxGroups(SoftmaxGroups):
    """
    Implements a multi-label softmax output layer with inhibition across the various softmax groups.
    """

    def __init__(self, dim_in, num_groups, num_classes, membership):
        """
        Initialize fields of the multi-label softmax layer w/ inhibition.

        Parameters
        ----------
        See SoftmaxGroups class for others...

        membership : ndarray (num_groups x num_classes)
          Inhibition group membership for each neuron (0 to ignore neuron)
        """

        super().__init__(dim_in, num_groups, num_classes)

        # Flatten the array of membership
        self.membership = membership.flatten()

    def forward(self, feats):
        """
        Perform the main processing steps for the softmax groups w/ inhibition.

        Parameters
        ----------
        feats : Tensor (B x T x F)
          Input features for a batch of tracks
          B - batch size,
          T - number of time steps (frames),
          E - dimensionality of input features

        Returns
        ----------
        tablature : Tensor (B x T x O)
          Tablature activations
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Acquire the activations for each neuron
        output = super().forward(feats)

        # Obtain the inhibition groups
        pitch_classes = torch.unique(self.membership[self.membership != 0])

        t = tools.get_current_time()

        # Loop through the inhibition groups
        # TODO - collapse loop if possible
        for pitch in pitch_classes:
            # Determine which string/fret combinations belong to the inhibition group
            string_fret_idcs = self.membership == pitch

            # Obtain the activations for the in-group string/fret combinations
            activations = output[..., string_fret_idcs]

            # Determine the maximum activation within the group
            max_val, _ = torch.max(activations, axis=-1)

            # Gate the maximum values at zero
            max_val = torch.nn.functional.relu(max_val)

            # Repeat the maximum values for each combination to vectorize the subtraction
            max_val = torch.cat([max_val.unsqueeze(-1)] * torch.sum(string_fret_idcs), axis=-1)

            # Subtract the maximum activation from in-group all combinations besides the max
            activations[activations < max_val] -= max_val[activations < max_val]

            # Replace the original activations for the in-group string/fret combinations
            output[..., string_fret_idcs] = activations

        tools.compute_time_difference(t, True, 'Inhibition')

        return output
