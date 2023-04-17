# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np


# -----------------------------------------------------------------------------

def aggregation_1d(data, agg_num, operation="sum"):
    """Aggregate one-dimensional array.

    Parameters
    ----------
    data : ndarray
        Array (one-dimensional) with data [arbitrary]
    agg_num : int
        Aggregation number [-]
    operation : str
        Aggregation operation ("sum" or "mean")

    Returns
    -------
    data_agg : ndarray
        Array (one-dimensional) with aggregated data [arbitrary]"""

    # Check input arguments
    if data.ndim != 1:
        raise TypeError("Input array must be one-dimensional")
    if data.size % agg_num != 0:
        raise TypeError("'agg_numb' is inconsistent with length of 'data'")
    if operation not in ("sum", "mean"):
        raise TypeError("unknown operation")
    if data.dtype == bool:
        print("Cast boolean array to type 'np.int32'")
        data = data.astype(np.int32)

    # Perform aggregation
    if operation == "sum":
        data_agg = np.sum(data.reshape(int(data.size / agg_num), agg_num),
                          axis=1)
    else:
        data_agg = np.mean(data.reshape(int(data.size / agg_num), agg_num),
                           axis=1)

    return data_agg


# -----------------------------------------------------------------------------

def aggregation_2d(data, agg_num_0, agg_num_1, operation="sum"):
    """Aggregate two-dimensional array.

    Parameters
    ----------
    data: ndarray
        Array (two-dimensional) with data [arbitrary]
    agg_num_0 : int
        Aggregation number along first dimension [-]
    agg_num_1 : int
        Aggregation number along second dimension [-]
    operation : str
        Aggregation operation ("sum" or "mean")

    Returns
    -------
    data_agg : ndarray
        Array (two-dimensional) with aggregated data [arbitrary]"""

    # Check input arguments
    if data.ndim != 2:
        raise TypeError("Input array must be two-dimensional")
    if (data.shape[0] % agg_num_0 != 0) or (data.shape[1] % agg_num_1 != 0):
        raise TypeError("'agg_numb' is inconsistent with shape of 'data'")
    if operation not in ("sum", "mean"):
        raise TypeError("unknown operation")
    if data.dtype == bool:
        print("Cast boolean array to type 'np.int32'")
        data = data.astype(np.int32)

    # Perform aggregation
    y = np.arange(0, data.shape[0], agg_num_0)
    temp = np.add.reduceat(data, y, axis=0, dtype=data.dtype)
    x = np.arange(0, data.shape[1], agg_num_1)
    data_agg = np.add.reduceat(temp, x, axis=1, dtype=data.dtype)
    if operation == "mean":
        if np.issubdtype(data.dtype, np.integer):
            print("Cast integer array to type 'np.float32'")
            data_agg = data_agg.astype(np.float32)
        data_agg /= float(agg_num_0 * agg_num_1)

    return data_agg


# -----------------------------------------------------------------------------

def nanaverage(data_in, weights):
    """Compute weighted average from non-NaN-values.

    Parameters
    ----------
    data_in : ndarray
        Array (n-dimensional) with input data [arbitrary]
    weights : ndarray
        Array (n-dimensional) with weights [-]

    Returns
    -------
    data_out : float
        Value with weighted average [arbitrary]"""

    # Check input arguments
    if data_in.shape != weights.shape:
        raise TypeError("Inconsistent shapes of input arrays")
    if np.any(np.isnan(weights)):
        raise ValueError("Weight array contains NaN-value(s)")

    # Compute weighted average
    mask = ~np.isnan(data_in)
    data_out = np.average(data_in[mask], weights=weights[mask])

    return data_out


# -----------------------------------------------------------------------------

def bool_mask_extend(mask_in):
    """Extend 'True' region in two-dimensional boolean mask by one grid cell in
    every of the eight directions.

    Parameters
    ----------
    mask_in : ndarray of bool
        Array (two-dimensional) with input mask

    Returns
    -------
    mask_in : ndarray of bool
        Array (two-dimensional) with output mask"""

    # Check input arguments
    if mask_in.ndim != 2:
        raise TypeError("Input array must be two-dimensional")
    if mask_in.dtype != "bool":
        raise TypeError("'mask_in' must be boolean array")

    # Perform extension
    mask_in = mask_in.astype(np.int8)
    mask_out = mask_in.copy()
    mask_out[1:, :] += mask_in[:-1, :]  # shift up
    mask_out[:-1, :] += mask_in[1:, :]  # shift down
    mask_out[:, 1:] += mask_in[:, :-1]  # shift left
    mask_out[:, :-1] += mask_in[:, 1:]  # shift right
    mask_out[1:, 1:] += mask_in[:-1, :-1]  # shift up & right
    mask_out[:-1, :-1] += mask_in[1:, 1:]  # shift down & left
    mask_out[1:, :-1] += mask_in[:-1, 1:]  # shift up & left
    mask_out[:-1, 1:] += mask_in[1:, :-1]  # shift down & right

    return mask_out > 0


# -----------------------------------------------------------------------------

def consecutive_length_max(data, return_range_indices=False):
    """Compute maximal length of consecutive True values along the first
    dimension of a three-dimensional array. Optionally return the range (start
    and stop) indices of this sequence. In case sequence with the maximal
    length occurs multiple times, the indices of the first sequence is returned.

    Parameters
    ----------
    data : ndarray of bool
        Array (three-dimensional) with input data
    return_range_indices : bool, optional
        Option to return the range (start and stop) indices of the sequence

    Returns
    -------
    length_max : ndarray of int
        Array (two-dimensional) with maximal length of consecutive True
        values
    ind_start : ndarray of int, optional
        Array (two-dimensional) with start indices of the sequence. Is set to
        -1 if all values are False.
    ind_stop : ndarray of int, optional
        Array (two-dimensional) with stop indices of the sequence. Is set to
        -1 if all values are False."""

    # Check input arguments
    if len(data.shape) != 3:
        raise TypeError("Input arrays must be three-dimensional")
    if not issubclass(data.dtype.type, np.bool_):
        raise TypeError("Input arrays must be of boolean type")
    if not isinstance(return_range_indices, bool):
        raise TypeError("'return_range_indices' must be of boolean type")

    # Compute maximal consecutive length of True values
    count = np.zeros(data.shape[1:], dtype=np.int32)
    length_max = np.zeros(data.shape[1:], dtype=np.int32)
    ind_stop = np.zeros(data.shape[1:], dtype=np.int32)
    for i in range(data.shape[0]):
        count[data[i, :, :]] += 1
        count[~(data[i, :, :])] = 0
        mask = (count > length_max)
        length_max[mask] = count[mask]
        ind_stop[mask] = i
    ind_start = ind_stop - length_max + 1
    mask = (length_max == 0)
    ind_start[mask] = -1
    ind_stop[mask] = -1

    if not return_range_indices:
        return length_max
    else:
        return length_max, ind_start, ind_stop
