# My imports
from core.models import CausalBasicSoftmax

from amt_models.datasets import GuitarSet
from amt_models.features import VQT

from amt_models import train, validate
from amt_models.transcribe import *
from amt_models.evaluate import *

import amt_models.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join([CausalBasicSoftmax.model_name(),
                    GuitarSet.dataset_name(),
                    VQT.features_name()])

ex = Experiment('Onsets & Frames w/ VQT on GuitarSet w/ 6-fold Cross Validation')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 200

    # Number of training iterations to conduct
    iterations = 1000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 20

    # Number of samples to gather for a batch
    batch_size = 2

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 1

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    expr_cache = os.path.join(tools.constants.HOME, 'Desktop',
                              'guitar-multi-task', 'generated', 'experiments')
    root_dir = os.path.join(expr_cache, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def guitarset_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints,
                        batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Initialize the default guitar profile
    profile = tools.GuitarProfile()

    # Processing parameters
    dim_in = 192
    model_complexity = 3

    # Create the vqt data processing module
    data_proc = VQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([TablatureWrapper(profile=profile)])
    #validation_estimator = None

    # Initialize the evaluation pipeline
    validation_evaluator = ComboEvaluator({tools.KEY_LOSS : LossWrapper(),
                                           tools.KEY_MULTIPITCH : MultipitchEvaluator(),
                                           tools.KEY_TABLATURE : TablatureEvaluator(profile=profile)})

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Keep all cached data/features under amt_models/generated for now
    gset_cache = None

    # Initialize an empty dictionary to hold the average results across fold
    results = dict()

    # Perform each fold of cross-validation
    for k in range(6):
        # Determine the name of the split being removed
        test_hold_out = '0' + str(k)
        val_hold_out = '0' + str(5 - k)

        print('--------------------')
        print(f'Fold {test_hold_out}:')

        # Remove the hold out split to get the training partition
        train_splits = splits.copy()
        train_splits.remove(test_hold_out)
        train_splits.remove(val_hold_out)

        val_splits = [val_hold_out]
        test_splits = [test_hold_out]

        print('Loading training partition...')

        # Create a dataset corresponding to the training partition
        gset_train = GuitarSet(base_dir=None,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               data_proc=data_proc,
                               profile=profile,
                               num_frames=num_frames,
                               reset_data=reset_data,
                               save_loc=gset_cache)

        # Create a PyTorch data loader for the dataset
        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        print('Loading validation partition...')

        # Create a dataset corresponding to the validation partition
        gset_val = GuitarSet(base_dir=None,
                             splits=val_splits,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             data_proc=data_proc,
                             profile=profile,
                             store_data=True,
                             save_loc=gset_cache)

        print('Loading testing partition...')

        # Create a dataset corresponding to the training partition
        gset_test = GuitarSet(base_dir=None,
                              splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              data_proc=data_proc,
                              profile=profile,
                              split_notes=False,
                              save_loc=gset_cache)

        print('Initializing model...')

        # Initialize a new instance of the model
        model = CausalBasicSoftmax(dim_in, profile, data_proc.get_num_channels(), model_complexity, gpu_id)
        model.change_device()
        model.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        print('Training classifier...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Set validation patterns for training
        validation_evaluator.set_patterns(['loss', 'f1', 'tdr'])

        # Train the model
        model = train(model=model,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      iterations=iterations,
                      checkpoints=checkpoints,
                      log_dir=model_dir,
                      val_set=gset_val,
                      estimator=validation_estimator,
                      evaluator=validation_evaluator)

        print('Transcribing and evaluating test partition...')

        # Add a save directory to the evaluators and reset the patterns
        validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
        validation_evaluator.set_patterns(None)

        # Get the average results for the fold
        fold_results = validate(model, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

        # Add the results to the tracked fold results
        results = append_results(results, fold_results)

        # Reset the results for the next fold
        validation_evaluator.reset_results()

        # Log the fold results for the fold in metrics.json
        ex.log_scalar('Fold Results', fold_results, k)

    # Log the average results for the fold in metrics.json
    ex.log_scalar('Overall Results', average_results(results), 0)
