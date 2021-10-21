# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import CQT

from amt_tools.train import train
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

from models.symbolic_models import RecConvLogisticEstimator
from tablature.GuitarProTabs import DadaGP
from tablature.GuitarSetTabs import GuitarSetTabs

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join(['TA', 'Logistic', 'dadagp', '10E0', 'p500'])

ex = Experiment('Separate Tablature Prediction Experiment')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 2000

    # Number of training iterations to conduct
    iterations = 200000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 200

    # Number of samples to gather for a batch
    batch_size = 25

    # The initial learning rate
    learning_rate = 1.0

    # The id of the gpu to use, if available
    gpu_id = 1

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def train_tablature(sample_rate, hop_length, num_frames, iterations, checkpoints,
                    batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=22)

    # Create the data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=192,
                    bins_per_octave=24)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([TablatureWrapper(profile=profile)])

    # Initialize the evaluation pipeline
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           TablatureEvaluator(profile=profile),
                                           SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

    # Base directories
    #gpro_bsdir = os.path.join('/', 'mnt', 'bigstorage', 'data', 'DadaGP')
    #gset_bsdir = os.path.join('/', 'mnt', 'bigstorage', 'data', 'GuitarSet')

    # Keep all cached data/features here
    #gset_cache = os.path.join(gset_bsdir, 'precomputed')
    gset_cache = os.path.join('..', 'generated', 'data')

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    gpro_train = DadaGP(base_dir=None,
    #gpro_train = DadaGP(base_dir=gpro_bsdir,
                        splits=['train'],
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        num_frames=num_frames,
                        data_proc=data_proc,
                        profile=profile,
                        save_data=False,
                        store_data=False,
                        augment_notes=False)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=gpro_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    print('Loading validation partition...')

    # Create a dataset corresponding to the validation partition
    gpro_val = DadaGP(base_dir=None,
    #gpro_val = DadaGP(base_dir=gpro_bsdir,
                      splits=['val'],
                      hop_length=hop_length,
                      sample_rate=sample_rate,
                      num_frames=num_frames,
                      data_proc=data_proc,
                      profile=profile,
                      save_data=False,
                      store_data=False,
                      augment_notes=False)

    print('Loading testing partition...')

    # Create a dataset corresponding to the testing partition
    gset_test = GuitarSetTabs(base_dir=None,
    #gset_test = GuitarSetTabs(base_dir=gset_bsdir,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              data_proc=data_proc,
                              profile=profile,
                              reset_data=reset_data,
                              store_data=True,
                              save_loc=gset_cache)

    print('Initializing model...')

    matrix_path = os.path.join('..', 'generated', 'matrices', 'dadagp_no_aug_p500.npz')

    # Initialize a new instance of the model
    tablature_layer = RecConvLogisticEstimator(profile=profile,
                                              model_complexity=3,
                                              matrix_path=matrix_path,
                                              device=gpu_id)
    tablature_layer.change_device()
    tablature_layer.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adadelta(tablature_layer.parameters(), learning_rate)

    print('Training model...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Set validation patterns for training
    validation_evaluator.set_patterns(['loss', 'f1', 'tdr', 'acc'])

    # Train the model
    tablature_layer = train(model=tablature_layer,
                            train_loader=train_loader,
                            optimizer=optimizer,
                            iterations=iterations,
                            checkpoints=checkpoints,
                            log_dir=model_dir,
                            single_batch=True,
                            val_set=gpro_val,
                            estimator=validation_estimator,
                            evaluator=validation_evaluator)

    print('Transcribing and evaluating test partition...')

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the fold
    results = validate(tablature_layer, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results for the fold in metrics.json
    ex.log_scalar('Overall Results', results, 0)
