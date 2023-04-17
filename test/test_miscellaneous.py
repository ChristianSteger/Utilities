# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utilities.miscellaneous import aggregation_1d, aggregation_2d
from utilities.miscellaneous import nanaverage, bool_mask_extend
from utilities.miscellaneous import consecutive_length_max

mpl.style.use("classic")

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

###############################################################################
# Test function 'consecutive_length_max'
###############################################################################

# -----------------------------------------------------------------------------
# First example
# -----------------------------------------------------------------------------

data = np.ones((17, 7, 5), dtype=bool)
data[:, 4, 0] = True  # (17)
data[:, 4, 1] = False  # (0)
data[:, 4, 2] = [True, True, True, False, False, True, True, True, False,
                 False, True, True, True, True, True, True, True]  # (7)
data[:, 4, 3] = [True, True, True, True, False, True, True, True, True,
                 False, True, True, False, True, True, False, False]  # (4)
data[:, 4, 4] = [False, False, False, True, True, False, False, True, True,
                 True, False, True, False, True, True, False, False]  # (3)
length_max = consecutive_length_max(data)
print(length_max[4, 0])
print(length_max[4, 1])
print(length_max[4, 2])
print(length_max[4, 3])
print(length_max[4, 4])

# -----------------------------------------------------------------------------
# Second example
# -----------------------------------------------------------------------------

data = np.ones((13, 5, 11), dtype=bool)
data[:, 0, 0] = [True] * 13  # (13, 0, 12)
data[:, 0, 1] = [False] * 13  # (0, -1, -1)
data[:, 0, 2] = [True, True, True, False, False, True, True, True, True,
                 False, False, True, True]  # (4, 5, 8)
data[:, 0, 3] = [False, True, True, True, False, True, False, True, True,
                 True, False, False, False]  # (3, 1, 3)
data[:, 0, 4] = [True, False, True, True, True, True, True, True, True,
                 False, True, False, False]  # (7, 2, 8)
data[:, 0, 5] = [False, False, False, True, True, False, True, False, True,
                 False, True, True, True]  # (3, 10, 12)

length_max = consecutive_length_max(data)
length_max, ind_start, ind_stop \
    = consecutive_length_max(data, return_range_indices=True)
print(length_max[0, 0], ind_start[0, 0], ind_stop[0, 0])
print(length_max[0, 1], ind_start[0, 1], ind_stop[0, 1])
print(length_max[0, 2], ind_start[0, 2], ind_stop[0, 2])
print(length_max[0, 3], ind_start[0, 3], ind_stop[0, 3])
print(length_max[0, 4], ind_start[0, 4], ind_stop[0, 4])
print(length_max[0, 5], ind_start[0, 5], ind_stop[0, 5])

# -----------------------------------------------------------------------------
# Third example
# -----------------------------------------------------------------------------

num = 3547
data = np.random.choice([True, True, True, True, True, False], num) \
    .reshape(num, 1, 1)
# data[:133] = True
# data[-277:] = True

length_max, ind_start, ind_stop \
    = consecutive_length_max(data, return_range_indices=True)
print(length_max[0][0], ind_start[0][0], ind_stop[0][0])
print(np.all(data[ind_start[0][0]:(ind_stop[0][0] + 1), 0, 0]))

data_add = np.hstack(([False], data[:, 0, 0], [False])).astype(int)
indices_start, = np.where(np.diff(data_add) > 0)
indices_end, = np.where(np.diff(data_add) < 0)
lengths_max = indices_end - indices_start
ind = np.argmax(lengths_max)
print(lengths_max[ind], indices_start[ind], indices_end[ind] - 1)
