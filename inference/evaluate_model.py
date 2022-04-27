# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import load_inhibition_matrix, trim_inhibition_matrix
from guitar_transcription_inhibition.models import TabCNNLogistic, LogisticTablatureEstimator
from guitar_transcription_inhibition.metrics import FalseAlarmErrors, InhibitionLoss
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
import torch
import os

fold = 0
model_path = f'/home/rockstar/Desktop/guitar-transcription/generated/experiments/<experiment>/models/fold-{fold}/model-XX00.pt'

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)

profile = model.profile
num_strings = profile.get_num_dofs()
num_pitches = profile.num_pitches

if isinstance(model, TabCNNLogistic):
    # TODO - should make model.tablature_layer = model.dense[-1] explicit
    silence_activations = model.dense[-1].silence_activations
else:
    silence_activations = True

default_matrix = LogisticTablatureEstimator.initialize_default_matrix(profile, silence_activations)

standard_matrix = load_inhibition_matrix('../generated/matrices/dadagp_silence_p1.npz')
standard_matrix = trim_inhibition_matrix(standard_matrix, num_strings, num_pitches, silence_activations)
standard_matrix = tools.array_to_tensor(standard_matrix, model.device)

boosted_matrix = load_inhibition_matrix('../generated/matrices/dadagp_silence_p128.npz')
boosted_matrix = trim_inhibition_matrix(boosted_matrix, num_strings, num_pitches, silence_activations)
boosted_matrix = tools.array_to_tensor(boosted_matrix, model.device)

score_matrices = {
    'l_str' : default_matrix,
    'l_inh' : standard_matrix,
    'l_inh+' : boosted_matrix
}

sample_rate = 22050
hop_length = 512

# Create the data processing module
data_proc = CQT(sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=192,
                bins_per_octave=24)

# Initialize the estimation pipeline
validation_estimator = ComboEstimator([TablatureWrapper(profile)])

# Initialize the evaluation pipeline
validation_evaluator = ComboEvaluator([LossWrapper(),
                                       MultipitchEvaluator(),
                                       TablatureEvaluator(profile=profile),
                                       FalseAlarmErrors(profile=profile),
                                       InhibitionLoss(profile=profile,
                                                      matrices=score_matrices,
                                                      silence_activations=silence_activations),
                                       SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Define expected path for calculated features and ground-truth
features_gt_cache = os.path.join('..', 'generated', 'data')

##################################################
# GuitarSet                                      #
##################################################

# Create a dataset object for GuitarSet
gset_test = GuitarSet(base_dir=None,
                      splits=[f'0{fold}'],
                      hop_length=hop_length,
                      sample_rate=sample_rate,
                      data_proc=data_proc,
                      profile=profile,
                      reset_data=False,
                      store_data=False,
                      save_loc=features_gt_cache)

# Get the average results for GuitarSet
results = validate(model, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

# Print the average results
print('GuitarSet Results')
print(results)

# Reset the evaluator
validation_evaluator.reset_results()
