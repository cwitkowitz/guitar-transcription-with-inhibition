# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import MelSpec

from amt_tools.train import validate
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
import torch
import os

multipitch_model_path = '../../generated/experiments/TabCNNJointCustom_GuitarSet_MelSpec/models/fold-0/model-50.pt'

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

model = torch.load(multipitch_model_path, map_location=device)
model.change_device(gpu_id)

profile = model.profile

sample_rate = 22050
hop_length = 512

# Create the Mel spectrogram data processing module
data_proc = MelSpec(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_mels=192,
                    decibels=False,
                    center=False)

# Initialize the estimation pipeline
validation_estimator = None

# Initialize the evaluation pipeline
validation_evaluator = ComboEvaluator([MultipitchEvaluator(),])
                                       #NoteEvaluator(key=tools.KEY_NOTE_ON),
                                       #NoteEvaluator(offset_ratio=0.2, key=tools.KEY_NOTE_OFF)])

# Define expected path for calculated features and ground-truth
features_gt_cache = os.path.join('..', '..', 'generated', 'data')

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
