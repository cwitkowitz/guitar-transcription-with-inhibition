# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import matplotlib.pyplot as plt
import amt_tools.tools as tools

# Regular imports
import numpy as np
import os

# Construct a path for loading the inhibition matrix
save_path = os.path.join('..', '..', 'generated', 'inhibition_matrix_gset_fold_1.npz')

# Load the inhibition matrix
matrix = np.load(save_path)['inh']

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Determine the number of unique activations
num_activations = matrix.shape[-1]

# Obtain the indices where each string begins
ticks = num_activations / profile.get_num_dofs() * np.arange(profile.get_num_dofs())

# Obtain the tick labels
labels = tools.DEFAULT_GUITAR_TUNING

# Obtain the complement of the matrix
likelihood = 1 - matrix

# Create a new figure and get the current axis
fig = tools.initialize_figure()
ax = fig.gca()

# Plot the inhibition matrix
ax.imshow(likelihood, extent=[0, num_activations, num_activations, 0])
ax.set_title('Inhibition Matrix')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.grid(c='black')

# Open the figure and pause execution
plt.show()
