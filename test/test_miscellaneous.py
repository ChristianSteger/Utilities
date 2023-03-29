# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("classic")

# Paths to folders
root_IAC = os.getenv("HOME") + "/Dropbox/IAC/"

# Load required functions
sys.path.append(root_IAC + "Scripts/Functions/")
from miscellaneous import aggregation_1d, aggregation_2d
from miscellaneous import nanaverage, bool_mask_extend

###############################################################################
# Test function 'aggregation_1d'
###############################################################################

# Sum
data = np.random.random(35)
data_agg = aggregation_1d(data, agg_num=5, operation="sum")
dev_abs_max = np.abs(data_agg - data.reshape(7, 5).sum(axis=1)).max()
print("Maximal deviation: %.10f" % dev_abs_max)

# Mean
data_agg = aggregation_1d(data, agg_num=5, operation="mean")
dev_abs_max = np.abs(data_agg - data.reshape(7, 5).mean(axis=1)).max()
print("Maximal deviation: %.10f" % dev_abs_max)

###############################################################################
# Test function 'aggregation_2d'
###############################################################################

# Data and aggregation numbers
data = np.random.random(12 * 15).reshape(12, 15)
agg_num_0 = 3
agg_num_1 = 5

# Sum
data_agg = aggregation_2d(data, agg_num_0=agg_num_0, agg_num_1=agg_num_1,
                          operation="sum")
data_agg_check = np.ones_like(data_agg)
for i in range(data_agg.shape[0]):
    for j in range(data_agg.shape[1]):
        slic = (slice(i * agg_num_0, (i + 1) * agg_num_0),
                slice(j * agg_num_1, (j + 1) * agg_num_1))
        data_agg_check[i, j] = data[slic].sum()
dev_abs_max = np.abs(data_agg - data_agg_check).max()
print("Maximal deviation: %.10f" % dev_abs_max)

# Mean
data_agg = aggregation_2d(data, agg_num_0=agg_num_0, agg_num_1=agg_num_1,
                          operation="mean")
data_agg_check = np.ones_like(data_agg)
for i in range(data_agg.shape[0]):
    for j in range(data_agg.shape[1]):
        slic = (slice(i * agg_num_0, (i + 1) * agg_num_0),
                slice(j * agg_num_1, (j + 1) * agg_num_1))
        data_agg_check[i, j] = data[slic].mean()
dev_abs_max = np.abs(data_agg - data_agg_check).max()
print("Maximal deviation: %.10f" % dev_abs_max)

###############################################################################
# Test function 'nanaverage'
###############################################################################

data_in = np.random.random(12).reshape(3, 4)
weights = np.random.random(12).reshape(3, 4)

print(np.average(data_in.ravel(), weights=weights.ravel()))
print(nanaverage(data_in, weights=weights))

data_in[:3, 0] = np.nan

print(np.average(data_in.ravel(), weights=weights.ravel()))
print(nanaverage(data_in, weights=weights))

###############################################################################
# Test function 'bool_mask_extend'
###############################################################################

# -----------------------------------------------------------------------------
# First example
# -----------------------------------------------------------------------------

# Test
mask_in = np.zeros((10, 10), dtype=bool)
mask_in[5:7, 3:5] = True
mask_in[7, 7] = True
mask_in[0, 0] = True
mask_in[2:-2, -2:] = True
mask_out = bool_mask_extend(mask_in)

data_plot = mask_out.astype(np.int8)
data_plot += mask_in.astype(np.int8)

plt.figure()
plt.pcolormesh(data_plot)

# -----------------------------------------------------------------------------
# Second example
# -----------------------------------------------------------------------------

# Test
x, y, = np.meshgrid(np.arange(101), np.arange(101))
dist = np.sqrt((x - x[50, 50]) ** 2 + (y - y[50, 50]) ** 2)
mask_in = (dist <= 40.0)
mask_out = bool_mask_extend(mask_in)

data_plot = mask_out.astype(np.int8)
data_plot += mask_in.astype(np.int8)

plt.figure()
plt.pcolormesh(data_plot)
