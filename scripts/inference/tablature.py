from amt_tools.evaluate import ComboEvaluator, MultipitchEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, IterativeStackedNoteTranscriber
from amt_tools.inference import run_online

import amt_tools.tools as tools

import matplotlib.pyplot as plt

import torch
import jams
import sys

# Define path to model, audio, and ground-truth
model_path = '/home/rockstar/Desktop/guitar-transcription/generated/experiments/tablature.pt'
jams_path = '/home/rockstar/Desktop/Datasets/GuitarSet/annotation/00_BN1-129-Eb_solo.jams'

# Feature extraction parameters
sample_rate = 22050
hop_length = 512

# GPU to use
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}'
                      if torch.cuda.is_available() else 'cpu')

# Add the path to the model definitions
sys.path.insert(0, '..')

# Load the model
model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)
model.eval()

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Open up the JAMS data
jam = jams.load(jams_path)

# Get the total duration of the file
duration = tools.extract_duration_jams(jam)

# Get the times for the start of each frame
times = tools.get_frame_times(duration, sample_rate, hop_length)

# Load the ground-truth notes
stacked_notes_ref = tools.load_stacked_notes_jams(jams_path)

# Obtain the multipitch predictions
multipitch_ref = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes_ref, times, profile)
# Determine the ground-truth tablature
tablature_ref = tools.stacked_multi_pitch_to_tablature(multipitch_ref, profile)
# Collapse the multipitch array
multipitch_ref = tools.stacked_multi_pitch_to_multi_pitch(multipitch_ref)

# Construct the ground-truth dictionary
ground_truth = {tools.KEY_MULTIPITCH : multipitch_ref,
                tools.KEY_TABLATURE : tablature_ref,
                tools.KEY_NOTES : stacked_notes_ref}

# Compute the features
features = {tools.KEY_FEATS : multipitch_ref, tools.KEY_TIMES : times}

# Define the estimation pipeline
estimator = ComboEstimator([TablatureWrapper(profile=profile, stacked=True),
                            IterativeStackedNoteTranscriber(profile=profile),
                            TablatureWrapper(profile=profile)])

# Define the evaluation pipeline
# TODO - stacked multipitch evaluator
# TODO - stacked note evaluator
evaluator = ComboEvaluator([MultipitchEvaluator(),
                            TablatureEvaluator(profile=profile),
                            SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Perform inference offline
predictions = run_online(features, model, estimator)

# Extract the ground-truth and predicted tablature
stacked_notes_est = predictions[tools.KEY_NOTES]

# Transpose the estimated notes and un-batch them
stacked_notes_est = tools.apply_func_stacked_representation(stacked_notes_est, tools.transpose_batched_notes)
stacked_notes_est = tools.apply_func_stacked_representation(stacked_notes_est, tools.batched_notes_to_notes)

# Convert the ground-truth and predicted notes to frets
stacked_frets_est = tools.stacked_notes_to_frets(stacked_notes_est)
stacked_frets_ref = tools.stacked_notes_to_frets(stacked_notes_ref)

# Plot both sets of notes and add an appropriate title
fig_est = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_est = tools.plot_guitar_tablature(stacked_frets_est, fig=fig_est)
fig_est.suptitle('Estimated')

fig_ref = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_ref = tools.plot_guitar_tablature(stacked_frets_ref, fig=fig_ref)
fig_ref.suptitle('Reference')

# Evaluate the predictions and track the results
results = evaluator.get_track_results(predictions, ground_truth)

# Print results to the console
print(results)

plt.show()
