# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

from models.tablature_layers import LogisticTablatureEstimator
from inhibition_matrix import plot_inhibition_matrix

# Regular imports
import matplotlib.pyplot as plt

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=22)

# Initialize the default inhibition matrix
inhibition_matrix = LogisticTablatureEstimator.initialize_default_matrix(profile, False)

# Plot the inhibition matrix
_ = plot_inhibition_matrix(inhibition_matrix)

# Call this to stop execution
plt.show(block=True)
