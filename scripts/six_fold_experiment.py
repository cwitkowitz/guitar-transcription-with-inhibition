# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
#from amt_tools.models import TabCNN
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

from amt_tools.train import train
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

from inhibition.inhibition_matrix import InhibitionMatrixTrainer, plot_inhibition_matrix
from models.tabcnn_variants import TabCNN, TabCNNLogistic
from statistics import compute_balanced_class_weighting
from visualize import plot_logistic_activations

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import matplotlib.pyplot as plt

import torch
import os

EX_NAME = '_'.join(['GT', 'Logistic'])

ex = Experiment('TabCNN w/ Tablature Estimation on GuitarSet w/ 6-fold Cross Validation')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 25

    # Number of training iterations to conduct
    iterations = 500000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 500

    # Number of samples to gather for a batch
    batch_size = 240

    # The initial learning rate
    learning_rate = 3E-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # Flag to use one split for validation
    validation_split = True

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)
    #root_dir = os.path.join('/', 'home', 'rockstar', 'Desktop', 'guitar-transcription-experiments-final', EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def six_fold_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints,
                       batch_size, learning_rate, gpu_id, reset_data, validation_split,
                       seed, root_dir):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Processing parameters
    dim_in = 192
    model_complexity = 1

    # Create the data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([TablatureWrapper(profile=profile)])

    # Initialize the evaluation pipeline
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           TablatureEvaluator(profile=profile),
                                           SoftmaxAccuracy(key=tools.KEY_TABLATURE)])

    # Base directories
    #gset_bsdir = os.path.join('/', 'mnt', 'bigstorage', 'data', 'GuitarSet')

    # Keep all cached data/features here
    #gset_cache = os.path.join(gset_bsdir, 'precomputed')
    gset_cache = os.path.join('..', 'generated', 'data')

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Initialize an empty dictionary to hold the average results across fold
    results = dict()

    # Perform each fold of cross-validation
    for k in range(6):
        # Seed everything with the same seed
        tools.seed_everything(seed)

        # Determine the testing split for the fold
        test_hold_out = '0' + str(k)

        print('--------------------')
        print(f'Fold {test_hold_out}:')

        # Remove the testing split
        train_splits = splits.copy()
        train_splits.remove(test_hold_out)
        test_splits = [test_hold_out]

        if validation_split:
            # Determine the validation split for the fold
            val_hold_out = '0' + str(5 - k)
            # Remove the validation split
            train_splits.remove(val_hold_out)
            val_splits = [val_hold_out]

        print('Loading training partition...')

        # Create a dataset corresponding to the training partition
        gset_train = GuitarSet(base_dir=None,
        #gset_train = GuitarSet(base_dir=gset_bsdir,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=num_frames,
                               data_proc=data_proc,
                               profile=profile,
                               reset_data=reset_data,
                               save_loc=gset_cache)

        # Create a PyTorch data loader for the dataset
        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        print(f'Loading testing partition (player {test_hold_out})...')

        # Create a dataset corresponding to the testing partition
        gset_test = GuitarSet(base_dir=None,
        #gset_test = GuitarSet(base_dir=gset_bsdir,
                              splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=False,
                              save_loc=gset_cache)

        if validation_split:
            print(f'Loading validation partition (player {val_hold_out})...')

            # Create a dataset corresponding to the validation partition
            gset_val = GuitarSet(base_dir=None,
            #gset_val = GuitarSet(base_dir=gset_bsdir,
                                 splits=val_splits,
                                 hop_length=hop_length,
                                 sample_rate=sample_rate,
                                 num_frames=None,
                                 data_proc=data_proc,
                                 profile=profile,
                                 store_data=False,
                                 save_loc=gset_cache)
        else:
            # Validate on the test set
            gset_val = gset_test

        print('Initializing model...')

        matrix_path = os.path.join('..', 'generated', 'matrices', f'<inhibition_matrix>.npz')

        # Initialize a new instance of the model
        tabcnn = TabCNNLogistic(dim_in=dim_in,
                                profile=profile,
                                in_channels=data_proc.get_num_channels(),
                                model_complexity=model_complexity,
                                #matrix_path=matrix_path,
                                silence_activations=False,
                                device=gpu_id)
        #tabcnn = TabCNN(dim_in=dim_in,
        #                profile=profile,
        #                in_channels=data_proc.get_num_channels(),
        #                model_complexity=model_complexity,
        #                device=gpu_id)
        tabcnn.change_device()
        tabcnn.train()

        """
        # TODO - remove later
        gset_full_train = GuitarSet(base_dir=None,
        #gset_full_train = GuitarSet(base_dir=gset_bsdir,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               data_proc=data_proc,
                               profile=profile,
                               save_loc=gset_cache)
        weighting = compute_balanced_class_weighting(gset_full_train, profile, True)
        tabcnn.dense[-1].set_weights(weighting)
        """

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(tabcnn.parameters(), learning_rate)

        # Define the visualization function with the root directory
        def vis_fnc(model, i):
            # Define the base directory for saving images
            save_dir = os.path.join(root_dir, 'visualization', f'fold-{k}')

            # Construct a save path for the raw, softmax, and final activations
            raw_dir = os.path.join(save_dir, 'raw_activations')
            smax_dir = os.path.join(save_dir, 'softmax_activations')
            final_dir = os.path.join(save_dir, 'final_activations')

            # Make sure the save directories exists
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(smax_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # Indicate whether the raw activations will include a silent string activation
            silent_class = False

            # Initialize a matrix trainer for visualizing the pairwise likelihood of activations
            trainer = InhibitionMatrixTrainer(profile, silent_string=silent_class, root=10)

            # Determine the parameters of the tablature
            num_strings = profile.get_num_dofs()
            num_classes = profile.num_pitches + int(silent_class)

            # Loop through all tracks in the validation set
            for n, track_data in enumerate(gset_val):
                # Extract the track name to use for save paths
                track_name = track_data[tools.KEY_TRACK]

                # Convert the track data to Tensors and add a batch dimension
                track_data = tools.dict_unsqueeze(tools.dict_to_tensor(track_data))
                # Run the track through the model without completing pre-processing steps
                raw_predictions = model(model.pre_proc(track_data)[tools.KEY_FEATS])
                # Throw away tracked gradients so the predictions can be copied
                raw_predictions = tools.dict_detach(raw_predictions)

                # Copy the predictions and remove the batch dimension
                raw_activations = tools.dict_squeeze(deepcopy(raw_predictions), dim=0)
                # Extract the tablature and switch the time and string/fret dimensions
                raw_activations = raw_activations[tools.KEY_TABLATURE].T

                # Reshape the activations according to the separate softmax groups
                smax_activations = raw_activations.view(num_strings, num_classes, -1)
                # Apply the softmax function to the activations for each group
                smax_activations = torch.softmax(smax_activations, dim=-2)
                # Return the activations to their original shape
                smax_activations = smax_activations.view(num_strings * num_classes, -1)

                # Convert the raw activations and softmax activations to NumPy arrays
                raw_activations = tools.tensor_to_array(raw_activations)
                smax_activations = tools.tensor_to_array(smax_activations)

                # Add the softmax activations to the inhibition matrix trainer
                trainer.step(smax_activations)

                # Package the predictions and ground-truth together for post-processing
                output = {tools.KEY_OUTPUT: deepcopy(raw_predictions),
                          tools.KEY_TABLATURE: track_data[tools.KEY_TABLATURE].to(model.device)}
                # Obtain final predictions and remove the batch dimension
                final_activations = tools.dict_squeeze(model.post_proc(output), dim=0)
                # Extract the tablature activations and convert to NumPy array
                final_activations = tools.tensor_to_array(final_activations[tools.KEY_TABLATURE])
                # Convert the final tablature predictions to logistic activations
                final_activations = tools.tablature_to_logistic(final_activations, profile, silent_class)

                # Determine the times associate with each frame
                times = np.arange(raw_activations.shape[-1]) * hop_length / sample_rate

                # Construct a save path for the raw activations
                raw_path = os.path.join(raw_dir, f'{track_name}-checkpoint-{i}.jpg')

                # Create a figure for plotting
                fig = plt.figure(figsize=(16, 8))
                # Plot the raw activations and return the figure
                fig = plot_logistic_activations(raw_activations, times=times, fig=fig)
                # Bump up the aspect ratio
                fig.gca().set_aspect(0.09)
                # Save the figure to the specified path
                fig.savefig(raw_path, bbox_inches='tight')
                # Close the figure
                plt.close(fig)

                # Construct a save path for the softmax activations
                smax_path = os.path.join(smax_dir, f'{track_name}-checkpoint-{i}.jpg')

                # Create a figure for plotting
                fig = plt.figure(figsize=(16, 8))
                # Plot the softmax activations and return the figure
                fig = plot_logistic_activations(smax_activations, times=times, v_bounds=[0, 1], fig=fig)
                # Bump up the aspect ratio
                fig.gca().set_aspect(0.09)
                # Save the figure to the specified path
                fig.savefig(smax_path, bbox_inches='tight')
                # Close the figure
                plt.close(fig)

                # Construct a save path for the final activations
                final_path = os.path.join(final_dir, f'{track_name}-checkpoint-{i}.jpg')

                # Create a figure for plotting
                fig = plt.figure(figsize=(16, 8))
                # Plot the final activations and return the figure
                fig = plot_logistic_activations(final_activations, times=times, v_bounds=[0, 1], fig=fig)
                # Bump up the aspect ratio
                fig.gca().set_aspect(0.09)
                # Save the figure to the specified path
                fig.savefig(final_path, bbox_inches='tight')
                # Close the figure
                plt.close(fig)

            # Compute the pairwise activations
            pairwise_activations = trainer.compute_current_matrix()

            # Construct a save path for the pairwise weights
            inhibition_path = os.path.join(save_dir, 'inhibition', f'{track_name}-checkpoint-{i}.jpg')
            # Make sure the save directory exists
            os.makedirs(os.path.dirname(inhibition_path), exist_ok=True)

            # Create a figure for plotting
            fig = plt.figure(figsize=(10, 10))
            # Plot the inhibition matrix and return the figure
            fig = plot_inhibition_matrix(pairwise_activations, v_bounds=[0, 1], fig=fig)
            # Save the figure to the specified path
            fig.savefig(inhibition_path, bbox_inches='tight')
            # Close the figure
            plt.close(fig)

        # Visualize the filterbank before conducting any training
        vis_fnc(tabcnn, 0)

        print('Training model...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Set validation patterns for training
        validation_evaluator.set_patterns(['loss', 'f1', 'tdr', 'acc'])

        # Train the model
        tabcnn = train(model=tabcnn,
                       train_loader=train_loader,
                       optimizer=optimizer,
                       iterations=iterations,
                       checkpoints=checkpoints,
                       log_dir=model_dir,
                       val_set=gset_val,
                       estimator=validation_estimator,
                       evaluator=validation_evaluator,
                       vis_fnc=vis_fnc)

        print('Transcribing and evaluating test partition...')

        # Add a save directory to the evaluators and reset the patterns
        validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
        validation_evaluator.set_patterns(None)

        # Get the average results for the fold
        fold_results = validate(tabcnn, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

        # Add the results to the tracked fold results
        results = append_results(results, fold_results)

        # Reset the results for the next fold
        validation_evaluator.reset_results()

        # Log the fold results for the fold in metrics.json
        ex.log_scalar('Fold Results', fold_results, k)

    # Log the average results for the fold in metrics.json
    ex.log_scalar('Overall Results', average_results(results), 0)
