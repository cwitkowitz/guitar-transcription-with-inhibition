# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import load_inhibition_matrix
from guitar_transcription_inhibition.metrics import FalseAlarmErrors, \
                                                    InhibitionLoss
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
import warnings
import datetime
import torch
import json
import os

# Construct the path to the top-level directory of the experiment
experiment_dir = os.path.join(tools.HOME, 'Desktop', 'guitar-transcription-with-inhibition',
                              'generated', 'experiments', '<EXPERIMENT>')

# Define the model checkpoints to use for six-fold cross-validation
checkpoints = [-1, -1, -1, -1, -1, -1]
# Tag to mark the metric used to choose checkpoints
identifier = 'tab'

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512
# Flag to re-acquire ground-truth data and re-calculate features
reset_data = False
# Choose the GPU on which to perform evaluation
gpu_id = 0

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Create a CQT feature extraction module spanning 8 octaves w/ 2 bins per semitone
data_proc = CQT(sample_rate=sample_rate, hop_length=hop_length, n_bins=192, bins_per_octave=24)

# Determine which directory under the experiment corresponds to the latest run
output_dir = sorted([dir for dir in os.listdir(experiment_dir) if dir.isdigit()])[-1]
# Specify the path to an output file to log results
output_path = os.path.join(experiment_dir, output_dir, f'six-fold-{identifier}.json')

# Initialize an empty dictionary to hold the average results across folds
results = dict()

# Loop through each fold
for k in range(6):
    # Obtain the path to the directory containing model checkpoints for the fold
    checkpoints_dir = os.path.join(experiment_dir, 'models', f'fold-{k}')

    if not os.path.exists(checkpoints_dir):
        # Move to the next fold if the fold directory doesn't exist
        continue

    # Parse the files in the directory to obtain all model checkpoint files
    model_paths = [path for path in os.listdir(checkpoints_dir) if 'model' in path]

    if not len(model_paths):
        # Move to the next fold if there are no model checkpoints
        continue

    # Sort the checkpoints by iteration
    model_paths = sorted(model_paths, key=tools.file_sort)
    # Determine the file name of the chosen checkpoint
    target_name = f'model-{checkpoints[k]}.pt'

    if target_name in model_paths:
        # If the checkpoint exits, build a path to it
        model_path = os.path.join(checkpoints_dir, target_name)
    else:
        if checkpoints[k] != -1:
            # If the checkpoint doesn't exist and the latest checkpoint wasn't chosen, throw a warning
            warnings.warn(f'Could not find file {target_name} under checkpoints directory for ' +
                          f'fold {k}. Choosing latest checkpoint instead.', category=RuntimeWarning)
        # Build a path to the latest checkpoint
        model_path = os.path.join(checkpoints_dir, model_paths[-1])

    # Determine which checkpoint ended up being selected
    checkpoint = int(os.path.basename(model_path).replace('model-', '').replace('.pt', ''))

    # Load the model onto the chosen device
    model = torch.load(model_path, map_location=device)
    model.change_device(gpu_id)

    # Initialize the estimation pipeline (Tablature -> Stacked Multi Pitch -> Multi Pitch)
    validation_estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                                           StackedMultiPitchCollapser(profile=model.profile)])

    # Load the standard and boosted inhibition matrices for computing inhibition loss
    standard_matrix = load_inhibition_matrix(os.path.join('..', 'generated', 'matrices', 'dadagp_p1.npz'))
    boosted_matrix = load_inhibition_matrix(os.path.join('..', 'generated', 'matrices', 'dadagp_p128.npz'))

    # Initialize the evaluation pipeline (Loss | Multi Pitch | Tablature | Distribution)
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           TablatureEvaluator(profile=model.profile),
                                           SoftmaxAccuracy(),
                                           FalseAlarmErrors(profile=model.profile),
                                           InhibitionLoss(profile=model.profile,
                                                          matrices={'l_inh_std': standard_matrix,
                                                                    'l_inh_plus': boosted_matrix},
                                                          silence_activations=model.tablature_layer.silence_activations
                                                          )])
    # Allocate the testing split for the fold
    test_splits = [GuitarSet.available_splits().pop(k)]

    # Define expected path for calculated features and ground-truth
    gset_cache = os.path.join('..', 'generated', 'data')

    # Create a dataset corresponding to the testing partition
    gset_test = GuitarSet(base_dir=None,
                          splits=test_splits,
                          hop_length=hop_length,
                          sample_rate=sample_rate,
                          num_frames=None,
                          data_proc=data_proc,
                          profile=model.profile,
                          reset_data=reset_data,
                          store_data=False,
                          save_loc=gset_cache)

    print(f'Evaluating fold {k} at checkpoint {checkpoint}...')

    # Compute the average results for the fold
    fold_results = validate(model, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

    with open(output_path, 'a') as json_file:
        # Add some other fields to the data before writing
        json_data = {'fold' : k,
                     'checkpoint' : checkpoint,
                     'time' : str(datetime.datetime.now()),
                     'results' : fold_results}
        # Write the fold results to the output file
        json.dump(json_data, json_file, sort_keys=True, indent=2)

    # Add the results to the tracked fold results
    results = append_results(results, fold_results)

with open(output_path, 'a') as json_file:
    # Add some other fields to the data before writing
    json_data = {'overall': {'time': str(datetime.datetime.now()),
                             'results': average_results(results)}}
    # Write the overall results to the output file
    json.dump(json_data, json_file, sort_keys=True, indent=2)
