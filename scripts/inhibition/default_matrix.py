# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

from inhibition_matrix import plot_inhibition_matrix

# Regular imports
import matplotlib.pyplot as plt
import numpy as np

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Extract tablature parameters
num_strings = profile.get_num_dofs()
num_pitches = profile.num_pitches

# Calculate output dimensionality
dim_out = num_strings * num_pitches

# Create a identity matrix with size equal to number of strings
inhibition_matrix = np.eye(num_strings)
# Repeat the matrix along both dimensions for each pitch
inhibition_matrix = np.repeat(inhibition_matrix, num_pitches, axis=0)
inhibition_matrix = np.repeat(inhibition_matrix, num_pitches, axis=1)
# Subtract out self-connections
inhibition_matrix = inhibition_matrix - np.eye(dim_out)

# Plot the inhibition matrix
_ = plot_inhibition_matrix(inhibition_matrix)

# Call this to stop execution
plt.show(block=True)
