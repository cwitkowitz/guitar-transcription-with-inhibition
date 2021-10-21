from amt_tools.evaluate import ComboEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedPitchListWrapper, IterativeStackedNoteTranscriber
from amt_tools.inference import run_single_frame
from amt_tools.features import MelSpec, AudioFileStream

import amt_tools.tools as tools

import torch
import time

# Define path to model, audio, and ground-truth
model_path = '/path/to/model'
audio_path = '/path/to/audio'
gt_path = '/path/to/ground-truth'

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
                            #StackedPitchListWrapper(profile=profile)])
                            IterativeStackedNoteTranscriber(profile=profile)])

# Define the evaluation pipeline
evaluator = ComboEvaluator([TablatureEvaluator(profile=profile),
                            SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Instantiate a dictionary to hold predictions
predictions = {}

# Number of frames of features to pad at beginning and end
feature_prime_amount = 4
feature_deprime_amount = 1

# Disable toolbar globally
tools.global_toolbar_disable()
# Create a figure to continually update
#visualizer = tools.StackedPitchListVisualizer(figsize=(10, 5), plot_frequency=10, time_window=5)
visualizer = tools.GuitarTablatureVisualizer(figsize=(10, 5), plot_frequency=12, time_window=5)

# Instantiate the audio stream and start streaming
feature_stream = AudioFileStream(data_proc, model.frame_width, audio_path, True, True)
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
        # Get the current time from the predictions
        current_time = new_predictions[tools.KEY_TIMES].item()
        # Get the current stacked note predictions
        stacked_notes = estimator.estimators[-1].get_active_stacked_notes(current_time)
        # Convert the stacked notes to frets
        stacked_frets = tools.stacked_notes_to_frets(stacked_notes, profile.tuning)
        # Call the visualizer's update loop
        visualizer.update(current_time, stacked_frets)
        #visualizer.update(current_time, new_predictions[tools.KEY_PITCHLIST])

# De-prime the buffer with features
for i in range(feature_deprime_amount):
    # Advance the buffer with empty frames
    features = feature_stream.buffer_empty_frame()
    # Perform inference on a single frame
    new_predictions = run_single_frame(features, model, estimator)
    # Append the new predictions
    predictions = tools.dict_append(predictions, new_predictions)
    # Get the current time from the predictions
    current_time = new_predictions[tools.KEY_TIMES].item()
    # Get the current stacked note predictions
    stacked_notes = estimator.estimators[-1].get_active_stacked_notes(current_time)
    # Convert the stacked notes to frets
    stacked_frets = tools.stacked_notes_to_frets(stacked_notes, profile.tuning)
    # Call the visualizer's update loop
    visualizer.update(current_time, stacked_frets)
    #visualizer.update(current_time, new_predictions[tools.KEY_PITCHLIST])

# Wait for 5 seconds before continuing
time.sleep(5)

# Stop and reset the feature stream
feature_stream.reset_stream()

# Evaluate the predictions and track the results
results = evaluator.get_track_results(predictions, ground_truth)

# Print results to the console
print(results)
