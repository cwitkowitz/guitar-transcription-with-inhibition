# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import CQT

import amt_tools.tools as tools

from inhibition_matrix import InhibitionMatrixTrainer
from tablature.GuitarProTabs import DadaGP

# Regular imports
import os

# Select z for taking the zth-root
z = 1000

# Construct a path for saving the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'matrices', f'dadagp_no_aug_r{z}.npz')

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Create the data processing module (only because TranscriptionDataset needs it)
# TODO - it would be nice to not need this -- see TODO in datasets/common.py
# Create the data processing module
data_proc = CQT(sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=192,
                bins_per_octave=24)

# Create a dataset using all of the GuitarPro tablature data
gpro_train = DadaGP(base_dir=None,
                    splits=['train'],
                    hop_length=hop_length,
                    sample_rate=sample_rate,
                    num_frames=None,
                    data_proc=data_proc,
                    profile=profile,
                    save_data=False,
                    store_data=False,
                    augment_notes=False)

# Obtain an inhibition matrix from the GuitarPro data
InhibitionMatrixTrainer(profile, gpro_train, save_path, root=z).train(residual_threshold=None)
