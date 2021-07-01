from amt_tools.transcribe import ComboEstimator, TablatureWrapper, IterativeStackedNoteTranscriber
from amt_tools.inference import run_single_frame
from amt_tools.features import MelSpec, AudioStream

import amt_tools.tools as tools

import torch

# Define path to model, audio, and ground-truth
model_path = '/home/rockstar/Desktop/guitar-transcription/generated/experiments/TabCNN_GuitarSet_MelSpec/models/fold-0/model-200.pt'

# Feature extraction parameters
sample_rate = 22050
hop_length = 512

# GPU to use
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}'
                      if torch.cuda.is_available() else 'cpu')

# Load the model
model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)
model.eval()

# Initialize the default guitar profile
profile = tools.GuitarProfile()

# Initialize the feature extraction protocol
data_proc = MelSpec(sample_rate=sample_rate, hop_length=hop_length, n_mels=192, decibels=False, center=False)

# Define the estimation pipeline
estimator = ComboEstimator([TablatureWrapper(profile=profile, stacked=True),
                            IterativeStackedNoteTranscriber(profile=profile)])

# Instantiate a dictionary to hold predictions
predictions = {}

# Instantiate the audio stream and start streaming
#feature_stream = AudioStream(data_proc, model.frame_width, audio, False, False)
# Fill the buffer with empties
feature_stream.prime_buffer(9)
# Start the feature stream
feature_stream.start_streaming()

while not feature_stream.query_finished():
    # Advance the buffer and get the current features
    features = feature_stream.buffer_new_frame()

    if feature_stream.query_buffer_full():
        # Perform inference on a single frame
        predictions = run_single_frame(features, model, predictions, estimator)
