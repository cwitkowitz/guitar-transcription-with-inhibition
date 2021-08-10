# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import matplotlib.pyplot as plt
import amt_tools.tools as tools

# Regular imports
import numpy as np
import os

# Construct a path for loading the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'inhibition_matrix.npz')

# Load the inhibition matrix
matrix = np.load(save_path)['inh']

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Determine the number of unique activations
num_activations = profile.get_num_dofs() * profile.num_pitches

# Obtain the indices where each string begins
ticks = profile.num_pitches * np.arange(profile.get_num_dofs())

# Obtain the tick labels
labels = tools.DEFAULT_GUITAR_TUNING

# Create a new figure and get the current axis
fig = tools.initialize_figure()
ax = fig.gca()

# Plot the inhibition matrix
ax.imshow(1 - matrix, extent=[0, num_activations, num_activations, 0])
ax.set_title('Inhibition Matrix')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.grid(c='black')

# Open the figure and pause execution
plt.show()
