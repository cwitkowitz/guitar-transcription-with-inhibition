# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

from amt_tools.train import validate
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
import torch
import os

import sys
sys.path.insert(0, '..')

from tablature.GuitarProTabs import DadaGP
from tablature.GuitarSetTabs import GuitarSetTabs

model_path = 'path/to/tablature/layer'

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)

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
validation_evaluator = ComboEvaluator([MultipitchEvaluator(),
                                       TablatureEvaluator(profile=profile),
                                       SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

# Define expected path for calculated features and ground-truth
features_gt_cache = os.path.join('..', '..', 'generated', 'data')

##################################################
# DadaGP                                         #
##################################################

"""
# Create a dataset object for GuitarSet
gpro_val = DadaGP(base_dir=None,
                  splits=['val'],
                  hop_length=hop_length,
                  sample_rate=sample_rate,
                  data_proc=data_proc,
                  profile=profile,
                  save_data=False,
                  store_data=False,
                  augment_notes=False)

# Get the average results for GuitarSet
results = validate(model, gpro_val, evaluator=validation_evaluator, estimator=validation_estimator)

# Print the average results
print('DadaGP Results')
print(results)

# Reset the evaluator
validation_evaluator.reset_results()
"""

##################################################
# GuitarSet                                      #
##################################################

# Create a dataset object for GuitarSet
gset_test = GuitarSetTabs(base_dir=None,
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
