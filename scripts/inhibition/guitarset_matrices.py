# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import MelSpec

import amt_tools.tools as tools

from .inhibition_matrix import InhibitionMatrixTrainer
from tablature.GuitarSetTabs import GuitarSetTabs

# Regular imports
import os

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Create the data processing module (only because TranscriptionDataset needs it)
# TODO - it would be nice to not need this -- see TODO in datasets/common.py
data_proc = MelSpec(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_mels=192,
                    decibels=False,
                    center=False)

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Get a list of the GuitarSet splits
splits = GuitarSetTabs.available_splits()

# Perform each fold of cross-validation
for k in range(6):
    # Determine the name of the splits being removed
    test_hold_out = '0' + str(k)

    print('--------------------')
    print(f'Fold {test_hold_out}:')

    # Remove the hold out splits to get the partitions
    train_splits = splits.copy()
    train_splits.remove(test_hold_out)

    # Construct a path for saving the inhibition matrix
    save_path = os.path.join('..', '..', 'generated', f'inhibition_matrix_guitarset_{test_hold_out}.npz')

    # Create a dataset using all of the GuitarSet tablature data, excluding the holdout fold
    gset_train = GuitarSetTabs(base_dir=None,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               data_proc=data_proc,
                               profile=profile,
                               num_frames=None,
                               save_data=False,
                               store_data=False,
                               augment_notes=True)

    # Obtain an inhibition matrix from the GuitarSet data
    InhibitionMatrixTrainer(profile, gset_train, save_path).train(residual_threshold=1E-2)
