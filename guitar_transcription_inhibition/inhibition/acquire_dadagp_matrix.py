# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import InhibitionMatrixTrainer
from guitar_transcription_inhibition.datasets import DadaGP

import amt_tools.tools as tools

# Regular imports
import os

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Flag to include silence associations
silence_activations = True

# Select the power for boosting
boost = 128

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Construct a path for saving the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'matrices', f'dadagp_p{boost}.npz')

# Create a dataset using all of the GuitarPro datasets data
dada_train = DadaGP(base_dir=None,
                    splits=['train', 'val'],
                    hop_length=hop_length,
                    sample_rate=sample_rate,
                    profile=profile,
                    num_frames=None,
                    save_data=False,
                    store_data=False,
                    max_duration=30)

# Obtain an inhibition matrix from the DadaGP data
InhibitionMatrixTrainer(profile=profile,
                        silence_activations=silence_activations,
                        boost=boost,
                        save_path=save_path).train(dada_train, residual_threshold=None)
