# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utilities.plot import truncate_colormap

mpl.style.use("classic")

###############################################################################
# Test function 'truncate_colormap'
###############################################################################

# Colormap
levels = np.arange(0.0, 5500.0, 500.0)
ticks = np.arange(0.0, 6000.0, 1000.0)
cmap = truncate_colormap(plt.get_cmap("terrain"), 0.3, 1.0)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)

# Test data
data = np.random.random((25, 25)) * 4500.0

# Plot
plt.figure()
plt.pcolormesh(data, cmap=cmap, norm=norm)
