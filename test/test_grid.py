# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.ops import transform
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cfeature
import fiona
from descartes import PolygonPatch
from pyproj import CRS, Transformer
from time import perf_counter
from utilities.grid import coord_edges, grid_frame, area_gridcells
from utilities.grid import polygon_inters_exact, polygon_inters_approx
from utilities.grid import polygon_rectangular

mpl.style.use("classic")

###############################################################################
# Test data
###############################################################################

# Create rotated latitude/longitude example coordinates (from EURO-CORDEX 0.11)
rlat_cent = np.linspace(-23.375, 21.835, 412)
rlon_cent = np.linspace(-28.375, 18.155, 424)
crs_rot_pole = ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25)

###############################################################################
# Test function 'coord_edges'
###############################################################################

# Test with artificial data
x_cent = [np.arange(4.5, 7.5, 0.5), np.arange(4.5, 7.5, 0.5)[::-1]]
y_cent = [np.arange(-0.75, 1.25, 0.25), np.arange(-0.75, 1.25, 0.25)[::-1]]

for i in range(len(x_cent)):

    x_edge, y_edge = coord_edges(x_cent[i], y_cent[i])

    # Output length of arrays and grid spacing
    print("Length x: " + str(len(x_cent[i])))
    print("Length grid_x: " + str(len(x_edge)))
    print("Grid spacing (x): " + "%.2f" % np.diff(x_cent[i]).mean())
    print("Length y: " + str(len(y_cent[i])))
    print("Length grid_y: " + str(len(y_edge)))
    print("Grid spacing (y): " + "%.2f" % np.diff(y_cent[i]).mean())

    # Plot
    plt.figure()
    plt.scatter(x_cent[i], np.ones(len(x_cent[i])), s=30, color="blue")
    plt.scatter(x_edge, np.ones(len(x_edge)), s=30, color="red")
    plt.ylim([0.9, 1.1])
    plt.title("x-coordinate")
    plt.figure()
    plt.scatter(y_cent[i], np.ones(len(y_cent[i])), s=30, color="blue")
    plt.scatter(y_edge, np.ones(len(y_edge)), s=30, color="red")
    plt.ylim([0.9, 1.1])
    plt.title("y-coordinate")

###############################################################################
# Test function 'grid_frame'
###############################################################################

# -----------------------------------------------------------------------------
# Test with artificial data
# -----------------------------------------------------------------------------

# Create grid coordinates
x_edge_1d = np.arange(17.)
y_edge_1d = np.arange(12.)
x_edge_2d, y_edge_2d = np.meshgrid(x_edge_1d, y_edge_1d)

# Plot
plt.figure()
ax = plt.axes()
plt.vlines(x=x_edge_1d, ymin=y_edge_1d.min(), ymax=y_edge_1d.max(),
           color="gray", lw=0.8)
plt.hlines(y=y_edge_1d, xmin=x_edge_1d.min(), xmax=x_edge_1d.max(),
           color="gray", lw=0.8)
for i in [0, 2, 5]:
    x_frame, y_frame = grid_frame(x_edge_1d, y_edge_1d, offset=i)
    poly = plt.Polygon(list(zip(x_frame, y_frame)), facecolor="none",
                       edgecolor="red", alpha=1.0, linewidth=2.5)
    ax.add_patch(poly)
    x_frame, y_frame = grid_frame(x_edge_2d, y_edge_2d, offset=i)
    poly = plt.Polygon(list(zip(x_frame, y_frame)), facecolor="none",
                       edgecolor="blue", alpha=1.0, linewidth=2.5,
                       linestyle="--")
    ax.add_patch(poly)
plt.axis([x_edge_1d[0] - 0.5, x_edge_1d[-1] + 0.5,
          y_edge_1d[0] - 0.5, y_edge_1d[-1] + 0.5])

# -----------------------------------------------------------------------------
# Test with real data
# -----------------------------------------------------------------------------

# Create grid coordinates
rlon_edge, rlat_edge = coord_edges(rlon_cent, rlat_cent)

# Plot
plt.figure()
ax = plt.axes()
plt.pcolormesh(rlon_edge, rlat_edge, np.ones((rlat_cent.size, rlon_cent.size)))
for i in [0, 50, 100]:
    rlon_frame, rlat_frame = grid_frame(rlon_edge, rlat_edge, offset=i)
    poly = plt.Polygon(list(zip(rlon_frame, rlat_frame)), facecolor="none",
                       edgecolor="black", alpha=1.0, linewidth=2.5)
    ax.add_patch(poly)
plt.axis([-30.0, 20.0, -25.0, 25.0])

###############################################################################
# Test function 'area_gridcells'
###############################################################################

# -----------------------------------------------------------------------------
# Simple test
# -----------------------------------------------------------------------------

x_edge_1d = np.cumsum(np.random.random(200))
y_edge_1d = np.cumsum(np.random.random(300))
test_0 = area_gridcells(x_edge_1d, y_edge_1d)
test_1 = area_gridcells(*np.meshgrid(x_edge_1d, y_edge_1d))

print(np.abs(test_0 - test_1).max())

# %timeit -r 1 -n 1 area_gridcells(x_edge_1d, y_edge_1d)
# %timeit -r 1 -n 1 area_gridcells(*np.meshgrid(x_edge_1d, y_edge_1d))

# -----------------------------------------------------------------------------
# Test with real data
# -----------------------------------------------------------------------------

# Map-projection (Equal Area Cylindrical projection)
crs_sphere = CRS.from_proj4("+proj=latlong +ellps=sphere")
crs_eac = CRS.from_proj4("+proj=cea +ellps=sphere")
# -> equal-area map projection required: cylindrical equal-area projection
x_edge, y_edge = Transformer.from_crs(crs_sphere, crs_eac) \
    .transform(*np.meshgrid(*coord_edges(rlon_cent, rlat_cent)))

# Calculate area of grid cells
area_gc = area_gridcells(x_edge, y_edge) / (1000. ** 2)  # [km2]

# Test plot
plt.figure()
plt.pcolormesh(rlon_cent, rlat_cent, area_gc)
plt.colorbar()

# -----------------------------------------------------------------------------
# Test for entire globe
# -----------------------------------------------------------------------------

# Map-projection (Equal Area Cylindrical projection)
d_grid = 0.1
lon_edge = np.linspace(-180.0, 180.0, 361)
lat_edge = np.linspace(-90.0, 90.0, 181)
lon_edge, lat_edge = np.meshgrid(lon_edge, lat_edge)
x_edge, y_edge = Transformer.from_crs(crs_sphere, crs_eac) \
    .transform(lon_edge, lat_edge)

# Compute grid cell areas
area_gc = area_gridcells(x_edge, y_edge) / (1000. ** 2)  # [km2]

# Compute surface area of Earth
rad_e = 6370997.0  # [m]
surf_area = (4 * np.pi * rad_e ** 2) / (1000. ** 2)  # [km2]

# Check area
dev = (area_gc.sum() / surf_area - 1.0) * 100.
print("Deviation in area: " + "%.5f" % dev + " %")

###############################################################################
# Test function 'polygon_inters'
###############################################################################

# -----------------------------------------------------------------------------
# Test with artificial data
# -----------------------------------------------------------------------------

# Compute grid cell area fractions
x_edge_1d = np.arange(0, 13)
y_edge_1d = np.arange(0, 11)
polygon = Polygon(zip([1.0, 5.0, 7.0, 8.0, 6.0, 6.0, 0.34, 2.234],
                      [1.0, 2.3234, 1.0, 3.0, 6.0, 5.0, 5.0, 4.0]))
area_frac_exact = polygon_inters_exact(*np.meshgrid(x_edge_1d, y_edge_1d),
                                       polygon, agg_cells=np.array([5, 2]))
area_frac_approx = polygon_inters_approx(x_edge_1d, y_edge_1d,
                                         polygon, num_samp=5)
# Colormap
levels = np.arange(0, 1.05, 0.05)
ticks = np.arange(0, 1.1, 0.1)
cmap = plt.get_cmap("YlGnBu")
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N)

# Plot
plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.05, wspace=0.10, width_ratios=[1, 1, 0.05])
ax = plt.subplot(gs[0, 0])
data_plot = np.ma.masked_where(area_frac_exact == 0.0, area_frac_exact)
plt.pcolormesh(data_plot, cmap=cmap, norm=norm)
poly_plot = PolygonPatch(polygon, facecolor="none",  edgecolor="black",
                         alpha=1.0, lw=2.0)
ax.add_patch(poly_plot)
plt.title("Exact method (area: %.3f" % np.sum(area_frac_exact) + ")",
          fontsize=12, fontweight="bold")
ax = plt.subplot(gs[0, 1])
data_plot = np.ma.masked_where(area_frac_approx == 0.0, area_frac_approx)
plt.pcolormesh(data_plot, cmap=cmap, norm=norm)
poly_plot = PolygonPatch(polygon, facecolor="none",  edgecolor="black",
                         alpha=1.0, lw=2.0)
ax.add_patch(poly_plot)
plt.title("Approximate method (area: %.3f" % np.sum(area_frac_approx) + ")",
          fontsize=12, fontweight="bold")
ax = plt.subplot(gs[0, 2])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                               ticks=ticks, orientation="vertical")

# Compare grid cell area fractions against polygon area
print("Area of the polygon: " + "%.5f" % polygon.area)
print("Area of the polygon (sum of exact area fractions): "
      + "%.5f" % np.sum(area_frac_exact))

# -----------------------------------------------------------------------------
# Test with real data
# -----------------------------------------------------------------------------
# -> Note: computation only exactly correct for Euclidean geometry

# Get borders of certain countries
countries = ("Switzerland", "Liechtenstein")
file_shp = shapereader.natural_earth("10m", "cultural", "admin_0_countries")
ds = fiona.open(file_shp)
geom_names = [i["properties"]["NAME"] for i in ds]
shp_geom = unary_union([shape(ds[geom_names.index(i)]["geometry"])
                        for i in countries])  # merge all polygons
crs_poly = CRS.from_string(ds.crs["init"])
ds.close()
# shp_geom = Polygon(shp_geom.exterior.simplify(0.1, preserve_topology=True))
# # optionally: simplify polygon

# Plot country borders
plt.figure()
ax = plt.axes()
poly_plot = PolygonPatch(shp_geom, facecolor="blue", edgecolor="black",
                         alpha=0.5)
ax.add_patch(poly_plot)
ax.autoscale_view()

# Transform polygon boundaries
project = Transformer.from_crs(crs_poly, CRS.from_user_input(crs_rot_pole),
                               always_xy=True).transform
shp_geom_trans = transform(project, shp_geom)

# Compute grid cell area fractions
rlon_edge, rlat_edge = coord_edges(rlon_cent, rlat_cent)
area_frac_exact = polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge),
                                       shp_geom_trans,
                                       agg_cells=np.array([10, 5, 2]))
area_frac_approx = polygon_inters_approx(rlon_edge, rlat_edge,
                                         shp_geom_trans,
                                         num_samp=5)

# Plot
map_ext = np.array([5.75, 10.70, 45.65, 47.9])  # [degree]
rad_earth = 6371.0  # approximate radius of Earth [km]
dist_x = 2.0 * np.pi * (rad_earth * np.cos(np.deg2rad(map_ext[2:].mean()))) \
         / 360.0 * (map_ext[1] - map_ext[0])
dist_y = 2.0 * np.pi * rad_earth / 360.0 * (map_ext[3] - map_ext[2])
fig = plt.figure(figsize=(10.0 + 0.3, 10.0 * (dist_y / dist_x)))
gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.05, wspace=0.05, width_ratios=[1, 0.03])
ax = plt.subplot(gs[0], projection=ccrs.PlateCarree())
ax.set_facecolor("lightgrey")
data_plot = np.ma.masked_where(area_frac_exact == 0.0, area_frac_exact)
# data_plot = np.ma.masked_where(area_frac_approx == 0.0, area_frac_approx)
plt.pcolormesh(rlon_edge, rlat_edge, data_plot, cmap=cmap, norm=norm,
               transform=crs_rot_pole)
bord_10m = cfeature.NaturalEarthFeature("cultural",
                                        "admin_0_boundary_lines_land",
                                        "10m",
                                        edgecolor="black",
                                        facecolor="none", zorder=1)
ax.add_feature(bord_10m)
gl = ax.gridlines(crs=ccrs.PlateCarree(),
                  xlocs=np.arange(0.0, 90.0, 0.5),
                  ylocs=np.arange(0.0, 90.0, 0.5),
                  linewidth=1, color="None", alpha=1.0, linestyle=":",
                  draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_aspect("auto")
poly_plot = PolygonPatch(shp_geom_trans, facecolor="none", edgecolor="black",
                         lw=2.5, transform=crs_rot_pole)
ax.add_patch(poly_plot)
ax.set_extent(map_ext, crs=ccrs.PlateCarree())
plt.title("Grid cell fraction within polygon [-]", fontsize=12,
          fontweight="bold", y=1.01)
ax = plt.subplot(gs[1])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                               ticks=ticks, orientation="vertical")
# fig.savefig(os.getenv("HOME") + "/Desktop/Grid_polygon_inters.png", dpi=300,
#             bbox_inches="tight")
# plt.close(fig)

# Check equality of different processing
area_frac_dp = \
    [polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge), shp_geom_trans),
     polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge), shp_geom_trans,
                          agg_cells=np.array([2])),
     polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge), shp_geom_trans,
                          agg_cells=np.array([5, 2])),
     polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge), shp_geom_trans,
                          agg_cells=np.array([10, 5, 2]))]
print(np.all(np.diff(np.concatenate([i[np.newaxis, :, :]
                                     for i in area_frac_dp],
                                    axis=0), axis=0) == 0.0))

# Check performance
t_beg = perf_counter()
af_0 = polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge), shp_geom_trans)
print("Elapsed time: %.2f" % (perf_counter() - t_beg) + " s")
t_beg = perf_counter()
af_1 = polygon_inters_exact(*np.meshgrid(rlon_edge, rlat_edge),
                            shp_geom_trans, agg_cells=np.array([10, 5, 2]))
print("Elapsed time: %.2f" % (perf_counter() - t_beg) + " s")
t_beg = perf_counter()
af_2 = polygon_inters_approx(rlon_edge, rlat_edge, shp_geom_trans, num_samp=5)
print("Elapsed time: %.2f" % (perf_counter() - t_beg) + " s")

###############################################################################
# Test function 'polygon_rectangular'
###############################################################################

box = (1.0, 2.0, 6.0, 5.0)
spacing = 0.25
polygon = polygon_rectangular(box, spacing)

# Plot
plt.figure()
ax = plt.axes()
x_corn = [box[0], box[2], box[2], box[0]]
y_corn = [box[1], box[1], box[3], box[3]]
plt.scatter(x_corn, y_corn, s=80, color="red")
poly_plot = PolygonPatch(polygon, facecolor="none",  edgecolor="blue",
                         alpha=0.2, lw=2.0)
x_poly, y_poly = polygon.exterior.coords.xy
plt.scatter(np.array(x_poly), np.array(y_poly), s=30, color="blue")
ax.add_patch(poly_plot)
coord_min = np.minimum(box[0], box[1]) - 0.3
coord_max = np.maximum(box[2], box[3]) + 0.3
plt.axis([coord_min, coord_max, coord_min, coord_max])
