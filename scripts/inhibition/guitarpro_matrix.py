# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import MelSpec

import amt_tools.tools as tools

from .inhibition_matrix import InhibitionMatrixTrainer
from tablature.GuitarProTabs import GuitarProTabs

# Regular imports
import os

# Construct a path for saving the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'inhibition_matrix_standard.npz')

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Create the data processing module (only because TranscriptionDataset needs it)
# TODO - it would be nice to not need this -- see TODO in datasets/common.py
data_proc = MelSpec(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_mels=192,
                    decibels=False,
                    center=False)

# Create a dataset using all of the GuitarPro tablature data
gpro_train = GuitarProTabs(base_dir=None,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           num_frames=2000,
                           data_proc=data_proc,
                           profile=profile,
                           save_data=False,
                           store_data=False,
                           max_duration=20,
                           augment_notes=True)

# Obtain an inhibition matrix from the GuitarPro data
InhibitionMatrixTrainer(profile, gpro_train, save_path).train(residual_threshold=1E-2)
