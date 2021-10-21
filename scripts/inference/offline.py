from amt_tools.evaluate import ComboEvaluator, MultipitchEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedNoteTranscriber
from amt_tools.inference import run_offline
from amt_tools.features import MelSpec

import amt_tools.tools as tools

import torch

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

# Initialize the default guitar profile
profile = tools.GuitarProfile()

# Initialize the CQT feature extraction protocol
data_proc = MelSpec(sample_rate=sample_rate, hop_length=hop_length, n_mels=192, decibels=False, center=False)

# Load the audio
audio, _ = tools.load_normalize_audio(audio_path, fs=sample_rate)

# Compute the features
features = {tools.KEY_FEATS : data_proc.process_audio(audio),
            tools.KEY_TIMES : data_proc.get_times(audio)}

# Define the estimation pipeline
estimator = ComboEstimator([TablatureWrapper(profile=profile, stacked=True),
                            StackedNoteTranscriber(profile=profile)])

# Define the evaluation pipeline
# TODO - stacked multipitch evaluator
# TODO - stacked note evaluator
evaluator = ComboEvaluator([TablatureEvaluator(profile=profile),
                            SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Perform inference offline
predictions = run_offline(features, model, estimator)

# Evaluate the predictions and track the results
results = evaluator.get_track_results(predictions, ground_truth)

# Print results to the console
print(results)
