# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from models.tablature_layers import LogisticTablatureEstimator
from inhibition_matrix_utils import plot_inhibition_matrix

import amt_tools.tools as tools

# Regular imports
from matplotlib import rcParams

import matplotlib.pyplot as plt

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Initialize the default inhibition matrix
matrix = LogisticTablatureEstimator.initialize_default_matrix(profile, False)

# Change the font and font size for the plot
rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 20

# Initialize a new figure with a large square shape
fig = plt.figure(figsize=(10, 10), tight_layout=True)

# Plot the inhibition matrix
_ = plot_inhibition_matrix(matrix, fig=fig)

# Call this to stop execution
plt.show(block=True)
