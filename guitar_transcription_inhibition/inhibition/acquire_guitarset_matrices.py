# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import InhibitionMatrixTrainer
from guitar_transcription_inhibition.datasets import GuitarSetTabs

import amt_tools.tools as tools

# Regular imports
import os


# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Flag to exclude validation split
validation_split = True
# Flag to include silence associations
silence_activations = True

# Select the power for boosting
boost = 1

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Keep all cached data/features here
gset_cache = os.path.join('..', '..', 'generated', 'data')

# Perform each fold of cross-validation
for k in range(6):
    # Allocate training/testing splits
    train_splits = GuitarSetTabs.available_splits()
    test_splits = [train_splits.pop(k)]

    if validation_split:
        # Allocate validation split
        val_splits = [train_splits.pop(k - 1)]

    print('--------------------')
    print(f'Fold {k}:')

    # Construct a path for saving the inhibition matrix
    save_path = os.path.join('..', '..', 'generated', 'matrices', f'guitarset_fold{k}_p{boost}.npz')

    # Create a dataset using all the GuitarSet tablature data, excluding holdout folds
    gset_train = GuitarSetTabs(base_dir=None,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               profile=profile,
                               num_frames=None,
                               save_loc=gset_cache)

    # Obtain an inhibition matrix from the GuitarSet data
    InhibitionMatrixTrainer(profile=profile,
                            silence_activations=silence_activations,
                            boost=boost,
                            save_path=save_path).train(gset_train, residual_threshold=None)
