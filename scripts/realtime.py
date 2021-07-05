from amt_tools.evaluate import ComboEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedPitchListWrapper, IterativeStackedNoteTranscriber
from amt_tools.inference import run_single_frame
from amt_tools.features import MelSpec, AudioStream

import amt_tools.tools as tools

import torch

# Define path to model, audio, and ground-truth
model_path = '/home/rockstar/Desktop/guitar-transcription/generated/experiments/TabCNN_GuitarSet_MelSpec/models/model-5000.pt'
audio_path = '/home/rockstar/Desktop/Datasets/GuitarSet/audio_mono-mic/00_BN1-129-Eb_solo_mic.wav'
gt_path = '/home/rockstar/Desktop/guitar-transcription/generated/data/GuitarSet/ground_truth/00_BN1-129-Eb_solo.npz'

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

# Initialize the feature extraction protocol
data_proc = MelSpec(sample_rate=sample_rate, hop_length=hop_length, n_mels=192, decibels=False, center=False)

# Define the estimation pipeline
estimator = ComboEstimator([TablatureWrapper(profile=profile, stacked=True),
                            StackedPitchListWrapper(profile=profile)])
                            #IterativeStackedNoteTranscriber(profile=profile)])

# Define the evaluation pipeline
evaluator = ComboEvaluator([TablatureEvaluator(profile=profile),
                            SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Load and normalize the audio
audio, _ = tools.load_normalize_audio(audio_path, fs=sample_rate, norm=-1)
# Instantiate a dictionary to hold predictions
predictions = {}

# Number of frames of features to pad at beginning and end
feature_prime_amount = 4
feature_deprime_amount = 1

# Disable toolbar globally
tools.global_toolbar_disable()
# Create a figure to continually update
visualizer = tools.StackedPitchListVisualizer(figsize=(10, 5),
                                              plot_frequency=10,
                                              time_window=4,
                                              colors=['red', 'green', 'black', 'red', 'green', 'black'],
                                              labels=tools.DEFAULT_GUITAR_LABELS)

# Instantiate the audio stream and start streaming
feature_stream = AudioStream(data_proc, model.frame_width, audio, True, True)
# Prime the buffer with empties
feature_stream.prime_frame_buffer(feature_prime_amount)
# Start the feature stream
feature_stream.start_streaming()

while not feature_stream.query_finished():
    # Advance the buffer and get the current features
    features = feature_stream.buffer_new_frame()

    if feature_stream.query_frame_buffer_full():
        # Perform inference on a single frame
        new_predictions = run_single_frame(features, model, estimator)
        # Append the new predictions
        predictions = tools.dict_append(predictions, new_predictions)
        # Call the visualizer's update loop
        visualizer.update(new_predictions[tools.KEY_TIMES].item(), new_predictions[tools.KEY_PITCHLIST])

# De-prime the buffer with features
for i in range(feature_deprime_amount):
    # Advance the buffer with empty frames
    features = feature_stream.buffer_empty_frame()
    # Perform inference on a single frame
    new_predictions = run_single_frame(features, model, estimator)
    # Append the new predictions
    predictions = tools.dict_append(predictions, new_predictions)
    # Call the visualizer's update loop
    visualizer.update(new_predictions[tools.KEY_TIMES].item(), new_predictions[tools.KEY_PITCHLIST])

# Stop and reset the feature stream
feature_stream.reset_stream()

# Evaluate the predictions and track the results
results = evaluator.get_track_results(predictions, ground_truth)

# Print results to the console
print(results)
