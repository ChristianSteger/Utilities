# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon


# -----------------------------------------------------------------------------

def truncate_colormap(cmap, val_min=0.0, val_max=1.0, num=100):
    """Truncate colormap to specific range between [0.0, 1.0].

    Parameters
    ----------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Input colormap
    val_min : float, optional
        Lower truncation value [0.0, 1.0] [-]
    val_max : float, optional
        Upper truncation value [0.0, 1.0] [-]
    num : int, optional
        Number of levels [-]

    Returns
    -------
    cmap_part : colormap object
        Truncated colormap object"""

    # Check input arguments
    if val_min >= val_max:
        raise TypeError("'val_min' must be smaller than 'val_max'")

    cmap_part = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({num},{a:.2f},{b:.2f})".format(
            num=cmap.name, a=val_min, b=val_max),
        cmap(np.linspace(val_min, val_max, num)))

    return cmap_part


# -----------------------------------------------------------------------------

def polygon2patch(polygon_shapely, **kwargs):
    """Convert Shapely (multi-)polygon to Matplotlib PatchCollection.

    Parameters
    ----------
    polygon_shapely : shapely.geometry.*.Polygon or MultiPolygon
        Input shapely geometry
    kwargs : various
        Additional arguments for matplotlib patch(es) (e.g. edgecolor,
        facecolor, linewidth, alpha)

    Returns
    -------
    patch_matplotlib : matplotlib.collections.PatchCollection
        Geometry as matplotlib patch collection"""

    # Check input arguments
    if not (isinstance(polygon_shapely, Polygon)
            or isinstance(polygon_shapely, MultiPolygon)):
        raise ValueError("Input is not a shapely polygon or multipolygon")

    # Convert shapely geometry
    if polygon_shapely.geom_type == "Polygon":  # Polygon
        path_geom = Path.make_compound_path(
            Path(np.asarray(polygon_shapely.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2])
              for ring in polygon_shapely.interiors])
        patch_geom = PathPatch(path_geom)
        patch_matplotlib = PatchCollection([patch_geom], **kwargs)
    else:  # MultiPolygon
        patch_geom = []
        for i in list(polygon_shapely.geoms):
            path_geom = Path.make_compound_path(
                Path(np.asarray(i.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in
                  i.interiors])
            patch_geom.append(PathPatch(path_geom))
        patch_matplotlib = PatchCollection(patch_geom, **kwargs)

    return patch_matplotlib
