# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ..datasets import GuitarSetTabs
from . import InhibitionMatrixTrainer
from amt_tools.features import CQT

import amt_tools.tools as tools

# Regular imports
import os

# Select the power for boosting
boost = 1

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Create the data processing module (only because TranscriptionDataset needs it)
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
    val_hold_out = '0' + str(5 - k)

    print('--------------------')
    print(f'Fold {test_hold_out}:')

    # Remove the hold out splits to get the partitions
    train_splits = splits.copy()
    train_splits.remove(test_hold_out)
    train_splits.remove(val_hold_out)

    # Construct a path for saving the inhibition matrix
    save_path = os.path.join('..', '..', 'generated', 'matrices', f'guitarset_{test_hold_out}_silence_p{boost}.npz')

    # Create a dataset using all of the GuitarSet datasets data, excluding the holdout fold
    gset_train = GuitarSetTabs(base_dir=None,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=None,
                               data_proc=data_proc,
                               profile=profile,
                               save_data=False,
                               store_data=False)

    # Obtain an inhibition matrix from the GuitarSet data
    InhibitionMatrixTrainer(profile, True, boost=boost, save_path=save_path).train(gset_train, residual_threshold=None)
