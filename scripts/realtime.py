from amt_tools.evaluate import ComboEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, IterativeStackedNoteTranscriber
from amt_tools.inference import run_single_frame
from amt_tools.features import CQT, AudioStream

import amt_tools.tools as tools

import torch

# Define path to model, audio, and ground-truth
model_path = '/home/rockstar/Desktop/amt-tools/generated/experiments/TabCNN_GuitarSet_CQT/models/fold-0/model-200.pt'
audio_path = '/home/rockstar/Desktop/Datasets/GuitarSet/audio_mono-mic/00_BN1-129-Eb_solo_mic.wav'
gt_path = '/home/rockstar/Desktop/amt-tools/generated/data/GuitarSet/ground_truth/00_BN1-129-Eb_solo.npz'

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

# Load the ground-truth
ground_truth = tools.load_unpack_npz(gt_path)

assert sample_rate == ground_truth[tools.KEY_FS]

# Initialize the default guitar profile
profile = tools.GuitarProfile()

# Initialize the CQT feature extraction protocol
data_proc = CQT(sample_rate=sample_rate, hop_length=hop_length, n_bins=192, bins_per_octave=24)

# Define the estimation pipeline
estimator = ComboEstimator([TablatureWrapper(profile=profile, stacked=True),
                            IterativeStackedNoteTranscriber(profile=profile)])

# Define the evaluation pipeline
evaluator = ComboEvaluator([TablatureEvaluator(profile=profile),
                            SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Load and normalize the audio
audio, _ = tools.load_normalize_audio(audio_path, fs=sample_rate, norm=-1)

# Instantiate a dictionary to hold predictions
predictions = {}

# Number of frames of features to pad at beginning and end
feature_pad_amount = model.frame_width // 2

# Instantiate the audio stream and start streaming
feature_stream = AudioStream(data_proc, model.frame_width, audio, True, False)
# Prime the buffer with empties
feature_stream.prime_buffer(feature_pad_amount)
# Start the feature stream
feature_stream.start_streaming()

while not feature_stream.query_finished():
    # Advance the buffer and get the current features
    features = feature_stream.buffer_new_frame()

    # TODO - remove the following line
    print(max(1, feature_stream.current_sample / len(feature_stream.audio)))

    if feature_stream.query_buffer_full():
        # Perform inference on a single frame
        predictions = run_single_frame(features, model, predictions, estimator)

# De-prime the buffer with features
for i in range(feature_pad_amount):
    # Advance the buffer with empty frames
    features = feature_stream.buffer_empty_frame()
    # Perform inference on a single frame
    predictions = run_single_frame(features, model, predictions, estimator)

# Stop and reset the feature stream
feature_stream.reset_stream()

# Evaluate the predictions and track the results
results = evaluator.get_track_results(predictions, ground_truth)

# Print results to the console
print(results)
