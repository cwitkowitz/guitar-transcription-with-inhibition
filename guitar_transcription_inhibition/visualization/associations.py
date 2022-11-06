# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import InhibitionMatrixTrainer, \
                                                       plot_inhibition_matrix
from guitar_transcription_inhibition.datasets import GuitarSetTabs

import amt_tools.tools as tools

# Regular imports
from matplotlib import rcParams

import matplotlib.pyplot as plt
import os


# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Flag to include silence associations
silence_activations = True

# Select the power for boosting
boost = 128

# Change the font and font size for the plot
rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 20

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Construct a path to the directory to save visualiations
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'associations')
# Make sure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# All data/features cached here
gset_cache = os.path.join('..', '..', 'generated', 'data')

# Initialize GuitarSet tablature to visualize associations
gset_full = GuitarSetTabs(base_dir=None,
                          hop_length=hop_length,
                          sample_rate=sample_rate,
                          profile=profile,
                          num_frames=None,
                          save_loc=gset_cache)

# Loop through the dataset
for track_data in gset_full:
    # Extract the tablature
    tablature = track_data[tools.KEY_TABLATURE]
    # Convert the tablature to logistic activations and switch the dimensions
    logistic = tools.tablature_to_logistic(tablature, profile, silence_activations)
    # Initialize an inhibition matrix trainer for the track
    t = InhibitionMatrixTrainer(profile, silence_activations=silence_activations, boost=128)
    # Feed the logistic activations into the trainer
    t.step(logistic)
    # Obtain pairwise likelihoods
    pairwise_activations = t.compute_current_matrix()

    # Construct a save path for the pairwise visualization
    inhibition_path = os.path.join(save_dir, f'{track_data[tools.KEY_TRACK]}.jpg')

    # Create a figure for plotting
    fig = tools.initialize_figure(figsize=(10, 10))
    # Plot the inhibition matrix and return the figure
    fig = plot_inhibition_matrix(pairwise_activations, v_bounds=[0, 1], fig=fig)
    # Save the figure to the specified path
    fig.savefig(inhibition_path, bbox_inches='tight')
    # Close the figure
    plt.close(fig)
