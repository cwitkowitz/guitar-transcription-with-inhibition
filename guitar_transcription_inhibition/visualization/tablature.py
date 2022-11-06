# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.datasets import GuitarSetTabs

import amt_tools.tools as tools

# Regular imports
import matplotlib.pyplot as plt
import os


# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Construct a path to the directory to save visualiations
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'tablature')
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
    # Extract the track name
    track_name = track_data[tools.KEY_TRACK]

    # Load the ground-truth notes from the JAMS data
    stacked_notes = tools.load_stacked_notes_jams(gset_full.get_jams_path(track_name))
    # Convert the ground-truth notes to frets
    stacked_frets = tools.stacked_notes_to_frets(stacked_notes)

    # Construct a save path for the pairwise weights
    tablature_path = os.path.join(save_dir, f'{track_name}.jpg')

    # Create a figure for plotting
    fig = tools.initialize_figure(figsize=(20, 5))
    # Plot the ground-truth notes and return the figure
    fig = tools.plot_guitar_tablature(stacked_frets, fig=fig)
    # Save the figure to the specified path
    fig.savefig(tablature_path, bbox_inches='tight')
    # Close the figure
    plt.close(fig)
