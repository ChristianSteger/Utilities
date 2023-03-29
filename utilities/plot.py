# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
import matplotlib as mpl

# -----------------------------------------------------------------------------


def truncate_colormap(cmap, val_min=0.0, val_max=1.0, num=100):
    """Truncate colormap to specific range between [0.0, 1.0].

    Parameters
    ----------
    cmap: matplotlib.colors.LinearSegmentedColormap
        Input colormap
    val_min: float
        Lower truncation value [0.0, 1.0] [-]
    val_max: float
        Upper truncation value [0.0, 1.0] [-]
    num: int
        Number of levels [-]

    Returns
    -------
    cmap_part: colormap object
        Truncated colormap object"""

    # Check input arguments
    if val_min >= val_max:
        raise TypeError("'val_min' must be smaller than 'val_max'")

    cmap_part = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({num},{a:.2f},{b:.2f})".format(
            num=cmap.name, a=val_min, b=val_max),
        cmap(np.linspace(val_min, val_max, num)))

    return cmap_part
