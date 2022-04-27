# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
import matplotlib.pyplot as plt
import jams

# TODO - add this to visualize script as a function? - if so, can use the function in inference scripts

# Define path to ground-truth
#jams_path = '/path/to/jams'
jams_path = '/home/rockstar/Desktop/Datasets/GuitarSet/annotation/01_SS2-88-F_comp.jams'
#00_BN2-166-Ab_solo
#01_SS2-88-F_comp
#03_SS1-100-C#_comp
#04_BN3-154-E_solo

# Feature extraction parameters
sample_rate = 22050
hop_length = 512

# Initialize a guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Open up the JAMS data
jam = jams.load(jams_path)

# Get the total duration of the file
duration = tools.extract_duration_jams(jam)

# Get the times for the start of each frame
times = tools.get_frame_times(duration, sample_rate, hop_length)

# Load the ground-truth notes
stacked_notes_ref = tools.load_stacked_notes_jams(jams_path)

# Obtain the multipitch predictions
multipitch_ref = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes_ref, times, profile)
# Determine the ground-truth datasets
tablature_ref = tools.stacked_multi_pitch_to_tablature(multipitch_ref, profile)
# Collapse the multipitch array
multipitch_ref = tools.stacked_multi_pitch_to_multi_pitch(multipitch_ref)

# Convert the ground-truth notes to frets
stacked_frets_ref = tools.stacked_notes_to_frets(stacked_notes_ref)

# Plot the ground-truth notes and add an appropriate title
fig_ref = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_ref = tools.plot_guitar_tablature(stacked_frets_ref, fig=fig_ref)
fig_ref.suptitle('Reference')

# Display the plots
plt.show()
