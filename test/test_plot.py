# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import shape
from cartopy.io import shapereader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fiona
from utilities.plot import truncate_colormap, polygon2patch
from utilities.plot import naturalearth_background
from utilities.plot import _get_path_data, _set_path_data
from utilities.plot import individual_background
from utilities.download import download_file

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

###############################################################################
# Test function 'polygon2patch'
###############################################################################

# -----------------------------------------------------------------------------
# Simple tests
# -----------------------------------------------------------------------------

# Polygon with hole (0)
exterior = [(1.0, 6.0), (7.0, 4.0), (8.0, 1.0), (1.0, 1.0), (1.0, 6.0)]
# clockwise
interior_0 = [(2.0, 4.0), (2.0, 2.0), (3.0, 2.0), (3.0, 4.0), (2.0, 4.0)]
interior_1 = [(4.0, 4.0), (4.0, 2.0), (7.0, 2.0), (4.0, 4.0)]
# anti-clockwise
polygon_0 = Polygon(exterior, holes=[interior_0, interior_1])

# Polygon (1)
exterior = [(9.0, 6.0), (12.0, 6.0), (9.0, 3.0), (9.0, 6.0)]
polygon_1 = Polygon(exterior)

# Polygon (2)
exterior = [(11.0, 4.0), (14.0, 5.0), (14.0, 1.0), (11.0, 1.0), (11.0, 4.0)]
interior = [(12.0, 3.0), (12.0, 2.0), (13.0, 2.0), (13.0, 4.0), (12.0, 3.0)]
polygon_2 = Polygon(exterior, holes=[interior])

# Plot individual polygons
plt.figure(figsize=(7, 4))
ax = plt.axes()
poly_plot = polygon2patch(polygon_0, facecolor="black", edgecolor="red",
                          lw=3.5, alpha=0.5)
ax.add_collection(poly_plot)
poly_plot = polygon2patch(polygon_1, facecolor="blue", edgecolor="black",
                          lw=1.5, ls="--", alpha=0.2)
ax.add_collection(poly_plot)
poly_plot = polygon2patch(polygon_2, facecolor="green", edgecolor="orange",
                          lw=2.5, ls=":", alpha=0.7)
ax.add_collection(poly_plot)
ax.axis([0.5, 14.5, 0.5, 6.5])

# Plot as collection
polygon_multi = MultiPolygon([polygon_0, polygon_1, polygon_2])
plt.figure(figsize=(7, 4))
ax = plt.axes()
poly_plot = polygon2patch(polygon_multi, facecolor="black", edgecolor="red",
                          lw=3.5, alpha=0.5)
ax.add_collection(poly_plot)
ax.axis([0.5, 14.5, 0.5, 6.5])

# -----------------------------------------------------------------------------
# Test with coordinate transformation
# -----------------------------------------------------------------------------

# Get borders of Italy (-> polygon has hole)
file_shp = shapereader.natural_earth("10m", "cultural", "admin_0_countries")
ds = fiona.open(file_shp)
geom_names = [i["properties"]["NAME"] for i in ds]
polygon_it = shape(ds[geom_names.index("Italy")]["geometry"])
ds.close()

# Plot in geographic coordinate system and map projection
coord_sys = (ccrs.PlateCarree(),
             ccrs.LambertConformal(central_longitude=13.0,
                                   central_latitude=42.0))
for i in coord_sys:
    plt.figure()
    ax = plt.axes(projection=i)
    poly_plot = polygon2patch(polygon_it, facecolor="black", edgecolor="blue",
                              lw=3.5, alpha=0.5, transform=ccrs.PlateCarree())
    ax.add_collection(poly_plot)
    ax.coastlines(resolution="10m")
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      xlocs=np.arange(0.0, 90.0, 1.0),
                      ylocs=np.arange(0.0, 90.0, 1.0),
                      linewidth=1, color="None", alpha=1.0, linestyle=":",
                      draw_labels=True, dms=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    plt.autoscale()

###############################################################################
# Test function 'naturalearth_background'
###############################################################################

# Default settings
crs_map = ccrs.PlateCarree()
plt.figure()
ax = plt.axes(projection=crs_map)
ax.set_extent((-13.0, 50.0, 30.0, 65.0), crs=crs_map)
naturalearth_background(ax)
ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)
ax.add_feature(cfeature.BORDERS, edgecolor="black", ls="-", lw=0.4)

# Orthographic projection
image_name = "cross_blended_hypso_with_relief_water_drains_and_ocean_bottom"
crs_map = ccrs.Orthographic(central_longitude=10.0, central_latitude=47.0)
plt.figure()
ax = plt.axes(projection=crs_map)
ax.set_global()
naturalearth_background(ax, image_name=image_name, image_res="medium",
                        interp_res=(4000, 4000))
ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)

# Rotated coordinate system
crs_map = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0)
fig = plt.figure()
ax = plt.axes(projection=crs_map)
ax.set_extent((-9.0, 8.0, -6.0, 7.0), crs=crs_map)
naturalearth_background(ax, image_name=image_name, image_res="high",
                        interp_res=(5000, 5000))
ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)
ax.add_feature(cfeature.BORDERS, edgecolor="black", ls="-", lw=0.4)
fig.savefig("/Users/csteger/Desktop/Alps.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Stereographic projection
crs_map = ccrs.Stereographic(central_latitude=90.0, central_longitude=0.0)
plt.figure()
ax = plt.axes(projection=crs_map)
ax.set_extent((-3500_000.0, 3000_000.0, -3000_000.0, 2500_000.0),
              crs=crs_map)
naturalearth_background(ax, image_name=image_name, image_res="high",
                        interp_res=(3000, 3000))
ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)
ax.add_feature(cfeature.BORDERS, edgecolor="black", ls="-", lw=0.4)

# Test all images
image_res = "low"  # "high", "medium", "low"
keys = ("cross_blended_hypso",
        "cross_blended_hypso_with_shaded_relief",
        "cross_blended_hypso_with_shaded_relief_and_water",
        "cross_blended_hypso_with_shaded_relief_water_and_drainages",
        "cross_blended_hypso_with_relief_water_drains_and_ocean_bottom",
        "natural_earth_i",
        "natural_earth_i_with_shaded_relief",
        "natural_earth_i_with_shaded_relief_and_water",
        "natural_earth_i_with_shaded_relief_water_and_drainages",
        "natural_earth_ii",
        "natural_earth_ii_with_shaded_relief",
        "natural_earth_ii_with_shaded_relief_and_water",
        "natural_earth_ii_with_shaded_relief_water_and_drainages",
        "ocean_bottom",
        "shaded_relief_basic",
        "gray_earth_with_shaded_relief_and_hypsography",
        "gray_earth_with_shaded_relief_hypsography_and_flat_water",
        "gray_earth_with_shaded_relief_hypsography_and_ocean_bottom",
        "gray_earth_with_shaded_relief_hypsography_ocean_bottom_and_drainages",
        "manual_shaded_relief")
for i in keys:
    print((" " + i + " ").center(79, "-"))
    try:
        crs_map = ccrs.LambertConformal(central_longitude=10.0,
                                        central_latitude=47.0)
        plt.figure()
        ax = plt.axes(projection=crs_map)
        ax.set_extent((-500_000.0, 500_000.0, -500_000.0, 500_000.0),
                      crs=crs_map)
        naturalearth_background(ax, image_name=i,
                                image_res=image_res, interp_res=(1000, 1000))
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)
        ax.add_feature(cfeature.BORDERS, edgecolor="black", ls="-", lw=0.4)
    except ValueError:
        print("Resolution not available for selected image")

# Check functions to set/get data path
# _get_path_data()
# _set_path_data("/Users/csteger/Desktop")

###############################################################################
# Test function 'individual_background'
###############################################################################

# Create temporary folder for images
path_images = "/Users/csteger/Desktop/Temp/"
os.mkdir(path_images)

# Download images automatically
# (source: https://visibleearth.nasa.gov/collection/1484/blue-marble)
files_url = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/147000/147190/"
    + "eo_base_2020_clean_geo.tif",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73963/"
    + "gebco_08_rev_bath_3600x1800_color.jpg",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73963/"
    + "gebco_08_rev_bath_21600x10800.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
    + "world.topo.bathy.200412.3x5400x2700.png")
for file_url in files_url:
    download_file(file_url, path_images)

# Download images manually
# (source: https://neo.gsfc.nasa.gov)
# - Sea Surface Temperature (MODIS, 2002+), PNG, 3600 x 1800 -> SST.png
# - Blue Marble: Next Generation, PNG, 3600 x 1800 -> Blue_Marble.png

PIL.Image.MAX_IMAGE_PIXELS = 233280000

# -----------------------------------------------------------------------------
# Greyscale image
# -----------------------------------------------------------------------------

# Load image
file_image = path_images + "gebco_08_rev_bath_21600x10800.png"
image = plt.imread(file_image)
print(image.shape)
print(np.max(image))
image = (image[:, :, 0] * 255).astype(np.uint8)  # scale values to [0, 255]

# Plot image
crs_map = ccrs.Orthographic(central_longitude=10.0, central_latitude=47.0)
plt.figure()
ax = plt.axes(projection=crs_map)
ax.set_global()
individual_background(ax, image, interp_res=(3000, 3000))
ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)

# -----------------------------------------------------------------------------
# Colour image
# -----------------------------------------------------------------------------

# Image information (scaling, drop array)
images = {
    "Blue_Marble.png": (True, False),
    "SST.png": (True, True),
    "eo_base_2020_clean_geo.tif": (False, False),
    "gebco_08_rev_bath_3600x1800_color.jpg": (False, False),
    "world.topo.bathy.200412.3x5400x2700.png": (True, False)
}

# Loop through images
for i in images:

    # Load image
    image = plt.imread(path_images + i)
    if images[i][0]:
        image = (image * 255).astype(np.uint8)  # scale values to [0, 255]
    if images[i][1]:
        image = image[:, :, :-1]  # drop array with constant values

    # Plot image
    crs_map = ccrs.Orthographic(central_longitude=10.0, central_latitude=47.0)
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=crs_map)
    ax.set_global()
    individual_background(ax, image, interp_res=(3000, 3000))
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", ls="-", lw=0.8)
    print("Image " + i + " plotted")
