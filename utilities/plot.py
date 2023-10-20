# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon
import cartopy
import cartopy.crs as ccrs
import PIL
import pyinterp
import zipfile
import textwrap
import utilities
from utilities.download import download_file


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


# -----------------------------------------------------------------------------

def _get_path_data():
    """Get the current data path and initialise it in case of non-existence.

    Returns
    -------
    path : str
        Current data path"""

    # Text file with data path
    path_utilities = os.path.dirname(utilities.__file__) + "/"
    file = path_utilities + "path_data.txt"

    # Initial definition of path
    if not os.path.isfile(file):
        path = ""
        while not os.path.isdir(path):
            txt = "Data path not yet defined. Please provide a valid path " \
                  + "to store Natural Earth raster images:"
            print(textwrap.fill(txt, 79))
            path = input()
        path = os.path.join(path, "")
        with open(file, "w") as f:
            f.write(path)

    # Get path
    else:
        with open(file, "r") as f:
            path = f.readline()

    return path


def _set_path_data(path):
    """Change or initialise data path.

    Parameters
    -------
    path : str
        New data path"""

    # Check input arguments
    if not os.path.isdir(path):
        raise ValueError("'path' does not exist")

    # Text file with data path
    path_utilities = os.path.dirname(utilities.__file__) + "/"
    file = path_utilities + "path_data.txt"

    # Initial definition of path
    path = os.path.join(path, "")
    if not os.path.isfile(file):
        print("Initial definition of path")
        with open(file, "w") as f:
            f.write(path)

    # Set path
    else:
        print("Overwrite existing path")
        with open(file, "w") as f:
            f.write(path)


def naturalearth_background(axis, image_name="shaded_relief_basic",
                            image_res="medium", interp_res=(1000, 1000)):
    """Add (high-resolution) Natural Earth raster feature to plot.

    Parameters
    ----------
    axis : cartopy.mpl.geoaxes.GeoAxes
        Cartopy axis
    image_name : str
        Name of Natural Earth raster image
    image_res : str
        Resolution of image ("low", "medium" or "high")
    interp_res : tuple of int
        Number of interpolation nodes in y- and x-direction"""

    # Dictionary of available Natural Earth raster images
    ne_raster = \
        {
            # -----------------------------------------------------------------
            # Cross Blended Hypsometric Tints (colour)
            # -----------------------------------------------------------------
            "cross_blended_hypsometric_tints": {
                "cross_blended_hypso":
                    {"high":   ("10m", "HYP_HR"),
                     "medium": ("10m", "HYP_LR")},
                "cross_blended_hypso_with_shaded_relief":
                    {"high":   ("10m", "HYP_HR_SR"),
                     "medium": ("10m", "HYP_LR_SR"),
                     "low":    ("50m", "HYP_50M_SR")},
                "cross_blended_hypso_with_shaded_relief_and_water":
                    {"high":   ("10m", "HYP_HR_SR_W"),
                     "medium": ("10m", "HYP_LR_SR_W"),
                     "low":    ("50m", "HYP_50M_SR_W")},
                "cross_blended_hypso_with_shaded_relief_water_and_drainages":
                    {"high":   ("10m", "HYP_HR_SR_W_DR"),
                     "medium": ("10m", "HYP_LR_SR_W_DR")},
                "cross_blended_hypso_with_relief_water_drains_"
                + "and_ocean_bottom":
                    {"high":   ("10m", "HYP_HR_SR_OB_DR"),
                     "medium": ("10m", "HYP_LR_SR_OB_DR")}
            },
            # -----------------------------------------------------------------
            # Natural Earth I (colour)
            # -----------------------------------------------------------------
            "natural_earth_i": {
                "natural_earth_i":
                    {"high":   ("10m", "NE1_HR_LC"),
                     "medium": ("10m", "NE1_LR_LC")},
                "natural_earth_i_with_shaded_relief":
                    {"high":   ("10m", "NE1_HR_LC_SR"),
                     "medium": ("10m", "NE1_LR_LC_SR")},
                    # "low":    ("50m", "NE1_50M_SR")}, not available online
                "natural_earth_i_with_shaded_relief_and_water":
                    {"high":   ("10m", "NE1_HR_LC_SR_W"),
                     "medium": ("10m", "NE1_LR_LC_SR_W"),
                     "low":    ("50m", "NE1_50M_SR_W")},
                "natural_earth_i_with_shaded_relief_water_and_drainages":
                    {"high":   ("10m", "NE1_HR_LC_SR_W_DR"),
                     "medium": ("10m", "NE1_LR_LC_SR_W_DR")}
            },
            # -----------------------------------------------------------------
            # Natural Earth II (colour)
            # -----------------------------------------------------------------
            "natural_earth_ii": {
                "natural_earth_ii":
                    {"high":   ("10m", "NE2_HR_LC"),
                     "medium": ("10m", "NE2_LR_LC")},
                "natural_earth_ii_with_shaded_relief":
                    {"high":   ("10m", "NE2_HR_LC_SR"),
                     "medium": ("10m", "NE2_LR_LC_SR"),
                     "low":    ("50m", "NE2_50M_SR")},
                "natural_earth_ii_with_shaded_relief_and_water":
                    {"high":   ("10m", "NE2_HR_LC_SR_W"),
                     "medium": ("10m", "NE2_LR_LC_SR_W"),
                     "low":    ("50m", "NE2_50M_SR_W")},
                "natural_earth_ii_with_shaded_relief_water_and_drainages":
                    {"high":   ("10m", "NE2_HR_LC_SR_W_DR"),
                     "medium": ("10m", "NE2_LR_LC_SR_W_DR")}
            },
            # -----------------------------------------------------------------
            # Ocean Bottom (colour)
            # -----------------------------------------------------------------
            "ocean_bottom": {
                "ocean_bottom":
                    {"medium": ("10m", "OB_LR"),
                     "low":    ("50m", "OB_50M")}
            },
            # -----------------------------------------------------------------
            # Bathymetry (colour)
            # -----------------------------------------------------------------
            # "bathymetry": {
            #     "bathymetry":
            #         {"low":    ("50m", "BATH_50M")}
            # },  # available online but in ".psd" format
            # -----------------------------------------------------------------
            # Shaded Relief (greyscale)
            # -----------------------------------------------------------------
            "shaded_relief": {
                "shaded_relief_basic":
                    {"high":   ("10m", "SR_HR"),
                     "medium": ("10m", "SR_LR"),
                     "low":    ("50m", "SR_50M")}
            },
            # -----------------------------------------------------------------
            # Gray Earth (greyscale)
            # -----------------------------------------------------------------
            "gray_earth": {
                "gray_earth_with_shaded_relief_and_hypsography":
                    {"high":   ("10m", "GRAY_HR_SR"),
                     "medium": ("10m", "GRAY_LR_SR"),
                     "low":    ("50m", "GRAY_50M_SR")},
                "gray_earth_with_shaded_relief_hypsography_and_flat_water":
                    {"high":   ("10m", "GRAY_HR_SR_W"),
                     "medium": ("10m", "GRAY_LR_SR_W"),
                     "low":    ("50m", "GRAY_50M_SR_W")},
                "gray_earth_with_shaded_relief_hypsography_and_ocean_bottom":
                    {"high":   ("10m", "GRAY_HR_SR_OB"),
                     "medium": ("10m", "GRAY_LR_SR_OB"),
                     "low":    ("50m", "GRAY_50M_SR_OB")},
                "gray_earth_with_shaded_relief_hypsography_ocean_bottom_"
                + "and_drainages":
                    {"high":   ("10m", "GRAY_HR_SR_OB_DR"),
                     "medium": ("10m", "GRAY_LR_SR_OB_DR")}
            },
            # -----------------------------------------------------------------
            # Manual Shaded Relief (greyscale)
            # -----------------------------------------------------------------
            "manual_shaded_relief": {
                "manual_shaded_relief":
                    {"low":    ("50m", "MSR_50M")}
            }
            # -----------------------------------------------------------------
        }

    # Formatted text with available names of images
    txt = ""
    for i in ne_raster.keys():
        txt += (" " + i + " ").center(79, "-") + "\n"
        for j in ne_raster[i].keys():
            txt += j + "\n"

    # Check input arguments
    if not isinstance(axis, cartopy.mpl.geoaxes.GeoAxes):
        raise TypeError("'axis' must be of type 'cartopy.mpl.geoaxes.GeoAxes'")
    keys = []
    category = []
    for i in ne_raster.keys():
        keys.extend(list(ne_raster[i].keys()))
        category.extend([i] * len(ne_raster[i].keys()))
    if image_name not in keys:
        raise ValueError("Invalid value for 'image_name'. Valid values are:\n"
                         + txt)
    if image_res not in ("low", "medium", "high"):
        raise ValueError("Invalid value for 'image_res'. Valid values are: "
                         + "low, medium, high")
    if image_res not in \
            ne_raster[category[keys.index(image_name)]][image_name].keys():
        raise ValueError("Resolution not available for selected image")
    if ((not isinstance(interp_res, tuple)) or (len(interp_res) != 2)
            or (any([not isinstance(i, int) for i in interp_res]))):
        raise TypeError("'interp_res' must be an integer tuple with length 2")
    if any([not (100 <= i <= 21_600) for i in interp_res]):
        raise ValueError("'interp_res' values must be between 100 and 21_600")

    # Download raster image
    image_info \
        = ne_raster[category[keys.index(image_name)]][image_name][image_res]
    if not os.path.isdir(_get_path_data() + image_info[1]):
        file_url = "https://www.naturalearthdata.com/http//" \
                   + "www.naturalearthdata.com/download/" + image_info[0] \
                   + "/raster/" + image_info[1] + ".zip"
        download_file(file_url, _get_path_data())
        file_zip = _get_path_data() + image_info[1] + ".zip"
        if len(zipfile.ZipFile(file_zip).namelist()[0].split("/")) == 1:
            path_out = file_zip[:-4]  # files are not in a subfolder
        else:
            path_out = _get_path_data()  # files are already in a subfolder
        with zipfile.ZipFile(file_zip, "r") as zip_ref:
            zip_ref.extractall(path_out)
        os.remove(file_zip)

    # Load image and create geographic coordinates
    PIL.Image.MAX_IMAGE_PIXELS = 233280000
    file_tif = _get_path_data() + image_info[1] + "/" + image_info[1] + ".tif"
    image = np.flipud(plt.imread(file_tif))  # (8100, 16200, (3)), RGB, 0-255
    extent = (-180.0, 180.0, -90.0, 90.0)
    dlon_h = (extent[1] - extent[0]) / (float(image.shape[1]) * 2.0)
    lon = np.linspace(extent[0] + dlon_h, extent[1] - dlon_h, image.shape[1])
    dlat_h = (extent[3] - extent[2]) / (float(image.shape[0]) * 2.0)
    lat = np.linspace(extent[2] + dlat_h, extent[3] - dlat_h, image.shape[0])
    crs_image = ccrs.PlateCarree()

    # Add data rows at poles (-90, +90 degree)
    lat_add = np.concatenate((np.array([-90.0]), lat, np.array([+90.0])))
    image_add = np.concatenate((image[:1, ...], image, image[-1:, ...]),
                               axis=0)
    x_axis = pyinterp.Axis(lon, is_circle=True)
    y_axis = pyinterp.Axis(lat_add)

    # Interpolate image to map
    extent_map = axis.axis()
    crs_map = axis.projection
    x_ip = np.linspace(extent_map[0], extent_map[1], interp_res[1])
    y_ip = np.linspace(extent_map[2], extent_map[3], interp_res[0])
    coord = crs_image.transform_points(crs_map, *np.meshgrid(x_ip, y_ip))
    lon_ip = coord[:, :, 0]
    lat_ip = coord[:, :, 1]
    mask = np.isfinite(lon_ip)
    # ------------------------------ colour image -----------------------------
    if image.ndim == 3:
        image_ip = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for i in range(3):
            grid = pyinterp.Grid2D(x_axis, y_axis, image_add[:, :, i]
                                   .transpose())
            data_ip = pyinterp.bivariate(
                grid, lon_ip[mask], lat_ip[mask],
                interpolator="bilinear", bounds_error=True, num_threads=0)
            image_ip[:, :, i][mask] = data_ip
    # ---------------------------- greyscale image ----------------------------
    else:
        image_ip = np.zeros(mask.shape, dtype=np.uint8)
        grid = pyinterp.Grid2D(x_axis, y_axis, image_add.transpose())
        data_ip = pyinterp.bivariate(
            grid, lon_ip[mask], lat_ip[mask],
            interpolator="bilinear", bounds_error=True, num_threads=0)
        image_ip[mask] = data_ip
    # -------------------------------------------------------------------------

    # Add image to axis
    if image.ndim == 3:
        axis.imshow(np.flipud(image_ip), extent=extent_map, transform=crs_map)
    else:
        axis.imshow(np.flipud(image_ip), extent=extent_map, transform=crs_map,
                    cmap="grey", vmin=0, vmax=255)


# -----------------------------------------------------------------------------

def individual_background(axis, image, interp_res=(1000, 1000)):
    """Add individual image to plot. The image must cover the entire globe in
    a Plate Carree projection.

    Parameters
    ----------
    axis : cartopy.mpl.geoaxes.GeoAxes
        Cartopy axis
    image : ndarray of uint8
        Image as two- or three-dimensional array with values between 0 and 255
        (y, x, (3: RGB))
    interp_res : tuple of int
        Number of interpolation nodes in y- and x-direction"""

    # Check input arguments
    if not isinstance(axis, cartopy.mpl.geoaxes.GeoAxes):
        raise TypeError("'axis' must be of type 'cartopy.mpl.geoaxes.GeoAxes'")
    if (not isinstance(image, np.ndarray)
            or (image.dtype != np.dtype("uint8"))
            or (not 2 <= image.ndim <= 3)):
        raise TypeError("'image' must be an array of type 'uint8' with "
                        + "2 or 3 dimensions")
    if ((not isinstance(interp_res, tuple)) or (len(interp_res) != 2)
            or (any([not isinstance(i, int) for i in interp_res]))):
        raise TypeError("'interp_res' must be an integer tuple with length 2")
    if any([not (100 <= i <= 21_600) for i in interp_res]):
        raise ValueError("'interp_res' values must be between 100 and 21_600")

    # Flip image
    image = np.flipud(image)

    # Create geographic coordinates
    extent = (-180.0, 180.0, -90.0, 90.0)
    dlon_h = (extent[1] - extent[0]) / (float(image.shape[1]) * 2.0)
    lon = np.linspace(extent[0] + dlon_h, extent[1] - dlon_h, image.shape[1])
    dlat_h = (extent[3] - extent[2]) / (float(image.shape[0]) * 2.0)
    lat = np.linspace(extent[2] + dlat_h, extent[3] - dlat_h, image.shape[0])
    crs_image = ccrs.PlateCarree()

    # Add data rows at poles (-90, +90 degree)
    lat_add = np.concatenate((np.array([-90.0]), lat, np.array([+90.0])))
    image_add = np.concatenate((image[:1, ...], image, image[-1:, ...]),
                               axis=0)
    x_axis = pyinterp.Axis(lon, is_circle=True)
    y_axis = pyinterp.Axis(lat_add)

    # Interpolate image to map
    extent_map = axis.axis()
    crs_map = axis.projection
    x_ip = np.linspace(extent_map[0], extent_map[1], interp_res[1])
    y_ip = np.linspace(extent_map[2], extent_map[3], interp_res[0])
    coord = crs_image.transform_points(crs_map, *np.meshgrid(x_ip, y_ip))
    lon_ip = coord[:, :, 0]
    lat_ip = coord[:, :, 1]
    mask = np.isfinite(lon_ip)
    # ------------------------------ colour image -----------------------------
    if image.ndim == 3:
        image_ip = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for i in range(3):
            grid = pyinterp.Grid2D(x_axis, y_axis, image_add[:, :, i]
                                   .transpose())
            data_ip = pyinterp.bivariate(
                grid, lon_ip[mask], lat_ip[mask],
                interpolator="bilinear", bounds_error=True, num_threads=0)
            image_ip[:, :, i][mask] = data_ip
    # ---------------------------- greyscale image ----------------------------
    else:
        image_ip = np.zeros(mask.shape, dtype=np.uint8)
        grid = pyinterp.Grid2D(x_axis, y_axis, image_add.transpose())
        data_ip = pyinterp.bivariate(
            grid, lon_ip[mask], lat_ip[mask],
            interpolator="bilinear", bounds_error=True, num_threads=0)
        image_ip[mask] = data_ip
    # -------------------------------------------------------------------------

    # Add image to axis
    if image.ndim == 3:
        axis.imshow(np.flipud(image_ip), extent=extent_map, transform=crs_map)
    else:
        axis.imshow(np.flipud(image_ip), extent=extent_map, transform=crs_map,
                    cmap="grey", vmin=0, vmax=255)
