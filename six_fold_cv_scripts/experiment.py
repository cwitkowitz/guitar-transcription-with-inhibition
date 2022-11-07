# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.metrics import FalseAlarmErrors, InhibitionLoss
from guitar_transcription_inhibition.inhibition import load_inhibition_matrix
from guitar_transcription_inhibition.models import TabCNNLogisticRecurrent
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

from amt_tools.train import train
from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               MultipitchEvaluator, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate, \
                               append_results, \
                               average_results

import amt_tools.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join(['Logistic', 'dadagp+', 'l10'])

ex = Experiment('Tablature Transcription on GuitarSet Logistic Variant w/ 6-fold Cross Validation')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 125

    # Number of training iterations to conduct
    iterations = 50000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 250

    # Number of samples to gather for a batch
    batch_size = 50

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate
    # features (useful if testing out different parameters)
    reset_data = False

    # Flag to set aside one split for validation
    validation_split = True

    # Multiplier for inhibition loss if applicable
    lmbda = 10

    # Path to inhibition matrix if applicable
    matrix_path = os.path.join('..', 'generated', 'matrices', 'dadagp_silence_p128.npz')

    # Flag to include an activation for silence in applicable output layers
    silence_activations = True

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment files
    root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def six_fold_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints, batch_size, gpu_id,
                       reset_data, validation_split, lmbda, matrix_path, silence_activations, seed, root_dir):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Create a CQT feature extraction module
    # spanning 8 octaves w/ 2 bins per semitone
    data_proc = CQT(sample_rate=sample_rate, hop_length=hop_length, n_bins=192, bins_per_octave=24)

    # Initialize the estimation pipeline (Tablature -> Stacked Multi Pitch -> Multi Pitch)
    validation_estimator = ComboEstimator([TablatureWrapper(profile=profile),
                                           StackedMultiPitchCollapser(profile=profile)])

    # Load the standard and boosted inhibition matrices for computing inhibition loss
    standard_matrix = load_inhibition_matrix(os.path.join('..', 'generated', 'matrices', 'dadagp_p1.npz'))
    boosted_matrix = load_inhibition_matrix(os.path.join('..', 'generated', 'matrices', 'dadagp_p128.npz'))

    # Initialize the evaluation pipeline (Loss | Multi Pitch | Tablature | Distribution)
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           TablatureEvaluator(profile=profile),
                                           SoftmaxAccuracy(),
                                           FalseAlarmErrors(profile=profile),
                                           InhibitionLoss(profile=profile,
                                                          matrices={'l_inh_std': standard_matrix,
                                                                    'l_inh_plus': boosted_matrix},
                                                          silence_activations=silence_activations)])

    # Keep all cached data/features here
    gset_cache = os.path.join('..', 'generated', 'data')

    # Initialize an empty dictionary to hold the average results across folds
    results = dict()

    # Perform six-fold cross-validation
    for k in range(6):
        print('--------------------')
        print(f'Fold {k}:')

        # Seed everything with the same seed
        tools.seed_everything(seed)

        # Set validation patterns for logging during training
        validation_evaluator.set_patterns(['loss', 'f1', 'tdr', 'acc', 'error', 'inh'])

        # Allocate training/testing splits
        train_splits = GuitarSet.available_splits()
        test_splits = [train_splits.pop(k)]

        if validation_split:
            # Allocate validation split
            val_splits = [train_splits.pop(k - 1)]

        print('Loading training partition...')

        # Create a dataset corresponding to the training partition
        gset_train = GuitarSet(base_dir=None,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=num_frames,
                               data_proc=data_proc,
                               profile=profile,
                               reset_data=(reset_data and k == 0),
                               save_loc=gset_cache)

        # Create a PyTorch data loader for the dataset
        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        if not validation_split:
            print(f'Loading testing partition (player {test_splits[0]})...')

        # Create a dataset corresponding to the testing partition
        gset_test = GuitarSet(base_dir=None,
                              splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=(not validation_split),
                              save_loc=gset_cache)

        if validation_split:
            print(f'Loading validation partition (player {val_splits[0]})...')

            # Create a dataset corresponding to the validation partition
            gset_val = GuitarSet(base_dir=None,
                                 splits=val_splits,
                                 hop_length=hop_length,
                                 sample_rate=sample_rate,
                                 num_frames=None,
                                 data_proc=data_proc,
                                 profile=profile,
                                 store_data=True,
                                 save_loc=gset_cache)
        else:
            # Perform validation on the testing partition
            gset_val = gset_test

        print('Initializing model...')

        # Initialize a new instance of the model
        tabcnn = TabCNNLogisticRecurrent(dim_in=data_proc.get_feature_size(),
                                         profile=profile,
                                         in_channels=data_proc.get_num_channels(),
                                         matrix_path=matrix_path,
                                         silence_activations=silence_activations,
                                         lmbda=lmbda,
                                         device=gpu_id)
        tabcnn.change_device()
        tabcnn.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adadelta(tabcnn.parameters(), lr=1.0)

        print('Training model...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Train the model
        tabcnn = train(model=tabcnn,
                       train_loader=train_loader,
                       optimizer=optimizer,
                       iterations=iterations,
                       checkpoints=checkpoints,
                       log_dir=model_dir,
                       val_set=gset_val,
                       estimator=validation_estimator,
                       evaluator=validation_evaluator)

        print(f'Transcribing and evaluating test partition (player {test_splits[0]})...')

        # Add a save directory to the evaluators
        validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
        # Reset the evaluation patterns to log everything
        validation_evaluator.set_patterns(None)

        # Compute the average results for the fold
        fold_results = validate(tabcnn, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

        # Add the results to the tracked fold results
        results = append_results(results, fold_results)

        # Reset the evaluators for the next fold
        validation_evaluator.reset_results()

        # Log the average results for the fold in metrics.json
        ex.log_scalar('Fold Results', fold_results, k)

    # Log the average results across all folds in metrics.json
    ex.log_scalar('Overall Results', average_results(results), 0)
