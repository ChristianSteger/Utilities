# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Polygon, MultiPolygon
from cartopy.io import shapereader
from shapely.geometry import shape
import cartopy.crs as ccrs
import fiona
from utilities.plot import truncate_colormap, polygon2patch

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
