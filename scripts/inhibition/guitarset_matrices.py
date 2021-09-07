# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import CQT

import amt_tools.tools as tools

from inhibition_matrix import InhibitionMatrixTrainer
from tablature.GuitarSetTabs import GuitarSetTabs

# Regular imports
import os

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Create the data processing module (only because TranscriptionDataset needs it)
# TODO - it would be nice to not need this -- see TODO in datasets/common.py
# Create the data processing module
data_proc = CQT(sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=192,
                bins_per_octave=24)

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
    save_path = os.path.join('..', '..', 'generated', 'matrices', f'guitarset_{test_hold_out}_no_aug_r5.npz')

    # Create a dataset using all of the GuitarSet tablature data, excluding the holdout fold
    gset_train = GuitarSetTabs(base_dir=None,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=None,
                               data_proc=data_proc,
                               profile=profile,
                               save_data=False,
                               store_data=False,
                               augment_notes=False)

    # Obtain an inhibition matrix from the GuitarSet data
    InhibitionMatrixTrainer(profile, gset_train, save_path, root=5).train(residual_threshold=None)
