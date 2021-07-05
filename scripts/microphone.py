from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedPitchListWrapper, IterativeStackedNoteTranscriber
from amt_tools.inference import run_single_frame
from amt_tools.features import MelSpec, MicrophoneStream

import amt_tools.tools as tools

import torch

# Define path to model, audio, and ground-truth
model_path = '/home/rockstar/Desktop/guitar-transcription/generated/experiments/TabCNN_GuitarSet_MelSpec/models/model-5000.pt'

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
estimator = ComboEstimator([TablatureWrapper(profile=profile, stacked=True),])
                            #StackedPitchListWrapper(profile=profile)])
                            #IterativeStackedNoteTranscriber(profile=profile)])

# Disable toolbar globally
tools.global_toolbar_disable()
# Create a figure to continually update
visualizer = tools.WaveformVisualizer(figsize=(10, 5), sample_rate=sample_rate)

# Instantiate the audio stream and start streaming
feature_stream = MicrophoneStream(data_proc, 1)#model.frame_width)
# Fill the buffer with empties
#feature_stream.prime_frame_buffer(9)
# Start the feature stream
feature_stream.start_streaming()

while not feature_stream.query_finished():
    # Advance the buffer and get the current features
    features = feature_stream.buffer_new_frame()

    """
    if feature_stream.query_frame_buffer_full():
        # Perform inference on a single frame
        predictions = run_single_frame(features, model, {}, estimator)
        visualizer.update(predictions[tools.KEY_TIMES], predictions[tools.KEY_PITCHLIST])
    """
    visualizer.update(features[tools.KEY_FEATS][0])
