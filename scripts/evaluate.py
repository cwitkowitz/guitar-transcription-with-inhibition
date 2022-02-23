# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from models.tabcnn_variants import TabCNNLogistic
from metrics import FalseAlarmErrors
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
import torch
import os

model_path = 'path/to/model'

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)

if isinstance(model, TabCNNLogistic):
    matrix_path = os.path.join('path', 'to', 'model')
    model.dense[-1].set_inhibition_matrix(matrix_path)

profile = model.profile

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
                                       SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Define expected path for calculated features and ground-truth
features_gt_cache = os.path.join('..', 'generated', 'data')

##################################################
# GuitarSet                                      #
##################################################

# Create a dataset object for GuitarSet
gset_test = GuitarSet(base_dir=None,
                      splits=['00'],
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
