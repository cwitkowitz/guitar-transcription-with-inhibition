# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import CQT

from amt_tools.train import train
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

from models.symbolic_models import BasicConv

from tablature.GuitarProTabs import DadaGP
from tablature.GuitarSetTabs import GuitarSetTabs

from inhibition.inhibition_matrix import InhibitionMatrixTrainer, plot_inhibition_matrix
from visualize import plot_logistic_activations

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import matplotlib.pyplot as plt

import torch
import os

EX_NAME = '_'.join(['TA', '6DSoftmax', 'BasicConv', 'test'])

ex = Experiment('Separate Tablature Prediction Experiment')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 500

    # Number of training iterations to conduct
    iterations = 1000000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 10000

    # Number of samples to gather for a batch
    batch_size = 100

    # The initial learning rate
    learning_rate = 1.0

    # The id of the gpu to use, if available
    gpu_id = 0

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
                      splits=['train'], # TODO - change back
                      hop_length=hop_length,
                      sample_rate=sample_rate,
                      num_frames=None,
                      data_proc=data_proc,
                      profile=profile,
                      save_data=False,
                      store_data=False,
                      augment_notes=False)

    # TODO - remove this later
    import random
    #random.shuffle(gpro_val.tracks)
    gpro_val.tracks = gpro_val.tracks[:100]
    #gpro_train.tracks = gpro_train.tracks[:100]

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
    tablature_layer = BasicConv(profile, 3, gpu_id)
    tablature_layer.change_device()
    tablature_layer.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adadelta(tablature_layer.parameters(), learning_rate)

    # Define the visualization function with the root directory
    def vis_fnc(model, i):
        # Define the base directory for saving images
        save_dir = os.path.join(root_dir, 'visualization')

        # Construct a save path for the raw, softmax, and final activations
        raw_dir = os.path.join(save_dir, 'raw_activations')
        smax_dir = os.path.join(save_dir, 'softmax_activations')
        final_dir = os.path.join(save_dir, 'final_activations')

        # Make sure the save directories exists
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(smax_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        # Indicate whether the raw activations will include a silent string activation
        silent_class = True

        # Initialize a matrix trainer for visualizing the pairwise likelihood of activations
        trainer = InhibitionMatrixTrainer(profile, silent_string=silent_class, root=10)

        # Determine the parameters of the tablature
        num_strings = profile.get_num_dofs()
        num_classes = profile.num_pitches + int(silent_class)

        # Loop through all tracks in the validation set
        for n, track_data in enumerate(gpro_val):
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

            """"""
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
            """"""

            # Construct a save path for the softmax activations
            smax_path = os.path.join(smax_dir, f'{track_name}-checkpoint-{i}.jpg')

            """"""
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
            """"""

            # Construct a save path for the final activations
            final_path = os.path.join(final_dir, f'{track_name}-checkpoint-{i}.jpg')

            """"""
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
            """"""

        # Compute the pairwise activations
        pairwise_activations = trainer.compute_current_matrix()

        # Construct a save path for the pairwise weights
        inhibition_path = os.path.join(save_dir, 'inhibition', f'{track_name}-checkpoint-{i}.jpg')
        # Make sure the save directory exists
        os.makedirs(os.path.dirname(inhibition_path), exist_ok=True)

        """"""
        # Create a figure for plotting
        fig = plt.figure(figsize=(10, 10))
        # Plot the inhibition matrix and return the figure
        fig = plot_inhibition_matrix(pairwise_activations, v_bounds=[0, 1], fig=fig)
        # Save the figure to the specified path
        fig.savefig(inhibition_path, bbox_inches='tight')
        # Close the figure
        plt.close(fig)
        """"""

    # Visualize the activations before conducting any training
    #vis_fnc(tablature_layer, 0)

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
                            evaluator=validation_evaluator,)
                            #vis_fnc=vis_fnc)

    print('Transcribing and evaluating test partition...')

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the fold
    results = validate(tablature_layer, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results for the fold in metrics.json
    ex.log_scalar('Overall Results', results, 0)
