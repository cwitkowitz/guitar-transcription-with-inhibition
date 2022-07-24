# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import InhibitionMatrixTrainer
from guitar_transcription_inhibition.datasets import DadaGP
from amt_tools.features import CQT

import amt_tools.tools as tools

# Regular imports
import os

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Create the data processing module (only because TranscriptionDataset needs it)
data_proc = CQT(sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=192,
                bins_per_octave=24)

# Create a dataset using all of the GuitarPro datasets data
gpro_train = DadaGP(base_dir=None,
                    splits=['train', 'val'],
                    hop_length=hop_length,
                    sample_rate=sample_rate,
                    num_frames=None,
                    data_proc=data_proc,
                    profile=profile,
                    save_data=False,
                    store_data=False,
                    max_duration=30)

# Select the power for boosting
boost = 128

# Construct a path for saving the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'matrices', f'dadagp_silence_p{boost}.npz')

# Obtain an inhibition matrix from the GuitarPro data
InhibitionMatrixTrainer(profile, True, boost=boost, save_path=save_path).train(gpro_train, residual_threshold=None)
