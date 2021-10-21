# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from inhibition_matrix import load_inhibition_matrix, plot_inhibition_matrix

# Regular imports
import matplotlib.pyplot as plt
import os

# Construct a path for loading the inhibition matrix
#save_path = os.path.join('path', 'to', 'matrix.npz')
save_path = os.path.join('..', '..', 'generated', 'matrices', 'dadagp_no_aug_p500.npz')
#save_path = os.path.join('..', '..', 'generated', 'matrices', 'guitarset_00_no_aug_r5.npz')

# Load the inhibition matrix
matrix = load_inhibition_matrix(save_path)

# Plot the inhibition matrix
_ = plot_inhibition_matrix(matrix)

# Call this to stop execution
plt.show(block=True)
