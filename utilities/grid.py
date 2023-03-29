# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from shapely.geometry import Polygon
import shapely
import shapely.vectorized


# -----------------------------------------------------------------------------

def coord_edges(x_cent, y_cent):
    """Compute edge coordinates from grid cell centre coordinates. Only works
    for grid with regular spacing.

    Parameters
    ----------
    x_cent: ndarray
        Array (one-dimensional) with x-coordinates of grid cell centres
        [arbitrary]
    y_cent: ndarray
        Array (one-dimensional) with y-coordinates of grid cell centres
        [arbitrary]

    Returns
    -------
    x_edge: ndarray
        Array (one-dimensional) with x-coordinates of grid cell edges
        [arbitrary]
    y_edge: ndarray
        Array (one-dimensional) with y-coordinates of grid cell edges
        [arbitrary]"""

    # Check input arguments
    if len(x_cent.shape) != 1 or len(y_cent.shape) != 1:
        raise TypeError("Input arrays must be one-dimensional")
    if (np.any(np.diff(np.sign(np.diff(x_cent))) != 0) or
            np.any(np.diff(np.sign(np.diff(y_cent))) != 0)):
        raise TypeError("Input arrays are not monotonically in- or decreasing")
    atol = 1e-05  # absolute tolerance
    if np.any(np.abs(np.diff(np.diff(x_cent))) > atol):
        raise ValueError("Irregular grid spacing in x-direction")
    if np.any(np.abs(np.diff(np.diff(y_cent))) > atol):
        raise ValueError("Irregular grid spacing in y-direction")

    # Compute grid coordinates
    dx_h = np.diff(x_cent).mean() / 2.0
    x_edge = np.hstack((x_cent[0] - dx_h,
                        x_cent[:-1] + np.diff(x_cent) / 2.,
                        x_cent[-1] + dx_h)).astype(x_cent.dtype)
    dy_h = np.diff(y_cent).mean() / 2.0
    y_edge = np.hstack((y_cent[0] - dy_h,
                        y_cent[:-1] + np.diff(y_cent) / 2.,
                        y_cent[-1] + dy_h)).astype(y_cent.dtype)

    return x_edge, y_edge


# -----------------------------------------------------------------------------

def grid_frame(x_edge, y_edge, offset=0):
    """Compute frame around a grid with a certain offset from the outer
    boundary.

    Parameters
    ----------
    x_edge: ndarray
        Array (one- or two-dimensional) with x-coordinates of grid cell edges
        [arbitrary]
    y_edge: ndarray
        Array (one- or two-dimensional) with y-coordinates of grid cell edges
        [arbitrary]
    offset: int
        offset value from outer boundary [-]

    Returns
    -------
    x_frame: ndarray
        Array (one-dimensional) with x-coordinates of frame [arbitrary]
    y_frame: ndarray
        Array (one-dimensional) with y-coordinates of frame [arbitrary]"""

    # Check input arguments
    if (x_edge.ndim != y_edge.ndim) or (y_edge.ndim > 2):
        raise TypeError("Dimensions of input arrays are unequal or larger"
                        + " than 2")
    if y_edge.ndim == 1:
        if not ((offset * 2 + 2) <= np.minimum(len(x_edge), len(y_edge))):
            raise TypeError("Offset value too large")
    else:
        if x_edge.shape != y_edge.shape:
            raise TypeError("Inconsistent shapes of input arrays")
        if not ((offset * 2 + 2) <= min(x_edge.shape)):
            raise TypeError("Offset value too large")

    # Compute frame from one-dimensional coordinate arrays
    if x_edge.ndim == 1:
        if offset > 0:
            x_edge = x_edge[offset:-offset]
            y_edge = y_edge[offset:-offset]
        x_frame = np.concatenate((x_edge[:-1],
                                  np.repeat(x_edge[-1], len(y_edge) - 1),
                                  x_edge[::-1][:-1],
                                  np.repeat(x_edge[0], len(y_edge) - 1)))
        y_frame = np.concatenate((np.repeat(y_edge[0], len(x_edge) - 1),
                                  y_edge[:-1],
                                  np.repeat(y_edge[-1], len(x_edge) - 1),
                                  y_edge[::-1][:-1]))

    # Compute frame from two-dimensional coordinate arrays
    else:
        if offset > 0:
            x_edge = x_edge[offset:-offset, offset:-offset]
            y_edge = y_edge[offset:-offset, offset:-offset]
        x_frame = np.hstack((x_edge[0, :],
                             x_edge[1:-1, -1],
                             x_edge[-1, :][::-1],
                             x_edge[1:-1, 0][::-1]))
        y_frame = np.hstack((y_edge[0, :],
                             y_edge[1:-1, -1],
                             y_edge[-1, :][::-1],
                             y_edge[1:-1, 0][::-1]))

    return x_frame, y_frame


# -----------------------------------------------------------------------------

def area_gridcells(x_edge, y_edge):
    """Compute area of grid cells. Assume plane (Euclidean) geometry.

    Parameters
    ----------
    x_edge: ndarray
        Array (one- or two-dimensional) with x-coordinates of grid cell edges
        [arbitrary]
    y_edge: ndarray
        Array (one- or two-dimensional) with y-coordinates of grid cell edges
        [arbitrary]

    Returns
    -------
    area_gc: ndarray
        Array (two-dimensional) with area of grid cells [arbitrary]"""

    # Check input arguments
    if ((x_edge.ndim not in [1, 2]) or (y_edge.ndim not in [1, 2]) or
            (x_edge.ndim != y_edge.ndim)):
        raise TypeError("Input arrays must be both either one- "
                        + "or two-dimensional")
    if x_edge.ndim == 2:
        if x_edge.shape != y_edge.shape:
            raise TypeError("Inconsistent shapes of input arrays")

    # Calculate areas for grid cells
    if x_edge.ndim == 1:  # regular grid
        area_gc = np.multiply(np.diff(x_edge).reshape(1, (len(x_edge) - 1)),
                              np.diff(y_edge).reshape((len(y_edge) - 1), 1))
    else:  # irregular grid
        area_gc = np.empty(np.asarray(x_edge.shape) - 1)
        for i in range(0, area_gc.shape[0]):
            for j in range(0, area_gc.shape[1]):
                x_vert = np.array([x_edge[i, j],
                                   x_edge[(i + 1), j],
                                   x_edge[(i + 1), (j + 1)],
                                   x_edge[i, (j + 1)]])
                y_vert = np.array([y_edge[i, j],
                                   y_edge[(i + 1), j],
                                   y_edge[(i + 1), (j + 1)],
                                   y_edge[i, (j + 1)]])
                polygon = Polygon(zip(x_vert, y_vert))
                area_gc[i, j] = polygon.area

    return area_gc


# -----------------------------------------------------------------------------

def polygon_inters_exact(x_edge, y_edge, polygon, agg_cells=np.array([])):
    """Compute area fractions of grid cells located inside the polygon.
    Exact method in which individual grid cells are intersected with polygon.
    Assume plane (Euclidean) geometry.

    Parameters
    ----------
    x_edge: ndarray
        Array (two-dimensional) with x-coordinates of grid cell edges
        [arbitrary]
    y_edge: ndarray
        Array (two-dimensional) with y-coordinates of grid cell edges
        [arbitrary]
    polygon: shapely.geometry.polygon.Polygon
        Shapely polygon [arbitrary]
    agg_cells: ndarray of int, optional
        Array with decreasing integers. The values determine the
        aggregation of grid cells into blocks for processing, which can
        decrease computational time considerably. Optimal values depend on the
        ratio between the size of the entire grid and the polygon as well as
        on the polygon's complexity.

    Returns
    -------
    area_frac: array_like
        Array (two-dimensional) with area fractions [-]"""

    # Check input arguments
    if len(x_edge.shape) != 2 or len(y_edge.shape) != 2:
        raise TypeError("Input arrays must be two-dimensional")
    if x_edge.shape != y_edge.shape:
        raise TypeError("Inconsistent shapes of input arrays")
    if not isinstance(polygon, shapely.geometry.polygon.Polygon):
        raise TypeError("'polygon' has incorrect type")
    if (agg_cells.size > 0) and \
            (not issubclass(agg_cells.dtype.type, np.integer)):
        raise TypeError("'agg_cells' must be an integer array")
    if np.any(np.diff(agg_cells) >= 0) or np.any(agg_cells <= 1):
        raise TypeError("'agg_cells' must be in descending order and contain "
                        + "values greater than one")

    # Allocate arrays
    shp = (x_edge.shape[0] - 1, x_edge.shape[1] - 1)
    area_frac = np.zeros(shp, dtype=np.float32)
    mask_proc = np.zeros(shp, dtype=bool)

    # Intersect 'blocks' of grid cells with polygon
    num_iter = 0
    for i in agg_cells:
        for j in range(0, shp[0], i):
            for k in range(0, shp[1], i):
                if not np.all(mask_proc[j:(j + i), k:(k + i)]):

                    ind_0 = np.minimum((j + i), shp[0])
                    ind_1 = np.minimum((k + i), shp[1])
                    x_box = np.array([x_edge[j, k],
                                      x_edge[ind_0, k],
                                      x_edge[ind_0, ind_1],
                                      x_edge[j, ind_1]])
                    y_box = np.array([y_edge[j, k],
                                      y_edge[ind_0, k],
                                      y_edge[ind_0, ind_1],
                                      y_edge[j, ind_1]])
                    box_poly = Polygon(zip(x_box, y_box))
                    
                    if polygon.contains(box_poly):
                        # polygon contains entire block
                        mask_proc[j:(j + i), k:(k + i)] = True
                        area_frac[j:(j + i), k:(k + i)] = 1.0
                    elif not polygon.intersects(box_poly):
                        # block entirely outside polygon
                        mask_proc[j:(j + i), k:(k + i)] = True
                    num_iter += 1

    # Intersect individual (remaining) grid cells with polygon
    for i, j in zip(*np.where(~mask_proc)):
        gc_x = np.array([x_edge[i, j],
                         x_edge[(i + 1), j],
                         x_edge[(i + 1), (j + 1)],
                         x_edge[i, (j + 1)]])
        gc_y = np.array([y_edge[i, j],
                         y_edge[(i + 1), j],
                         y_edge[(i + 1), (j + 1)],
                         y_edge[i, (j + 1)]])
        gc_poly = Polygon(zip(gc_x, gc_y))
        area_frac[i, j] = polygon.intersection(gc_poly).area / gc_poly.area
        num_iter += 1

    print("Number of steps: " + str(num_iter))

    return area_frac


# -----------------------------------------------------------------------------

def polygon_inters_approx(x_edge, y_edge, polygon, num_samp=1):
    """Compute area fractions of grid cells located inside the polygon.
    Approximate method in which intersecting areas are derived by checking
    points within the grid cells (one ore multiple sampling). Assume plane
    (Euclidean) geometry.

    Parameters
    ----------
    x_edge: ndarray
        Array (one- or two-dimensional) with x-coordinates of grid cell edges
        [arbitrary]
    y_edge: ndarray
        Array (one- or two-dimensional) with y-coordinates of grid cell edges
        [arbitrary]
    polygon: shapely.geometry.polygon.Polygon
        Shapely polygon [arbitrary]
    num_samp: int, optional
        Number of evenly distributed point-samples within a grid cell along
        one dimension. With e.g. 'num_samp = 5', 5 x 5 = 25 point locations are
        checked

    Returns
    -------
    area_frac: ndarray
        Array (two-dimensional) with area fractions [-]"""

    # Check input arguments
    if len(x_edge.shape) != 1 or len(y_edge.shape) != 1:
        raise TypeError("Input arrays must be one-dimensional")
    if not isinstance(polygon, shapely.geometry.polygon.Polygon):
        raise TypeError("'polygon' has incorrect type")
    if (num_samp < 1) or (num_samp > 100):
        raise ValueError("Sampling number must be in the range [1, 100]")

    # Intersect grid cells with polygon
    if num_samp == 1:  # no sub-sampling

        print("Sample centre of grid cells")
        x_cent = x_edge[:-1] + np.diff(x_edge) / 2.0
        y_cent = y_edge[:-1] + np.diff(y_edge) / 2.0
        area_frac = shapely.vectorized.contains(
            polygon, *np.meshgrid(x_cent, y_cent)).astype(np.float32)

    else:  # sub-sampling

        print("Sample " + str(num_samp ** 2) + " evenly distributed "
              + "locations within the grid cells")
        x_temp = np.linspace(x_edge[0], x_edge[-1],
                             (x_edge.size - 1) * num_samp + 1)
        xp = x_temp[0:None:num_samp]
        x = x_temp[:-1] + np.diff(x_temp) / 2.0
        x_samp = np.interp(x, xp, x_edge)
        y_temp = np.linspace(y_edge[0], y_edge[-1],
                             (y_edge.size - 1) * num_samp + 1)
        yp = y_temp[0:None:num_samp]
        y = y_temp[:-1] + np.diff(y_temp) / 2.0
        y_samp = np.interp(y, yp, y_edge)
        mask_bin = shapely.vectorized.contains(
            polygon, *np.meshgrid(x_samp, y_samp)).astype(np.float32)
        y_agg = np.arange(0, mask_bin.shape[0], num_samp)
        temp = np.add.reduceat(mask_bin, y_agg, axis=0)
        x_agg = np.arange(0, mask_bin.shape[1], num_samp)
        area_frac = np.add.reduceat(temp, x_agg, axis=1) \
            / (num_samp ** 2)

    return area_frac
