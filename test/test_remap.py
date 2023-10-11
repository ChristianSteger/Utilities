# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import subprocess
import cartopy.crs as ccrs
import urllib.request
import shutil
from utilities.grid import coord_edges, grid_frame
from utilities import remap

mpl.style.use("classic")

###############################################################################
# Create working directory and download test NetCDF file
###############################################################################

path_work = os.getenv("HOME") + "/Desktop/work/"
if not os.path.isdir(path_work):
    os.mkdir(path_work)

file_url = "https://confluence.ecmwf.int/download/attachments/140385202/" \
           + "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc?version=" \
           + "1&modificationDate=1591979822003&api=v2"
urllib.request.urlretrieve(file_url, path_work + "ERA5-Land_geopotential.nc")

###############################################################################
# Source grid: regular latitude/longitude grid
###############################################################################

# Get source grid information
file_in = path_work + "ERA5-Land_geopotential.nc"
ds = xr.open_dataset(file_in)
ds = ds.sel(longitude=slice(0.0, 40.0), latitude=slice(65.0, 30.0))
lon_cent_in = ds["longitude"].values
lat_cent_in = ds["latitude"].values
ds.close()
lon_edge_in, lat_edge_in = coord_edges(lon_cent_in, lat_cent_in)

# -----------------------------------------------------------------------------
# Target grid: regular latitude/longitude grid
# -----------------------------------------------------------------------------

# Define output grid
lon_cent = np.linspace(5.0, 20.0, 301)
lat_cent = np.linspace(42.0, 52.0, 201)
lon_edge, lat_edge = coord_edges(lon_cent, lat_cent)

# Check domain coverage
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
lon_frame, lat_frame = grid_frame(lon_edge_in, lat_edge_in, offset=0)
poly = plt.Polygon(list(zip(lon_frame, lat_frame)), facecolor="red",
                   edgecolor="red", alpha=0.2, linewidth=1.0)
ax.add_patch(poly)
lon_frame, lat_frame = grid_frame(lon_edge, lat_edge, offset=0)
poly = plt.Polygon(list(zip(lon_frame, lat_frame)), facecolor="none",
                   edgecolor="blue", alpha=1.0, linewidth=2.5)
ax.add_patch(poly)
ax.coastlines(resolution="50m", color="black", linewidth=1.0)
ax.axis([lon_cent_in[0] - 1.0, lon_cent_in[-1] + 1.0,
         lat_cent_in.min() - 1.0, lat_cent_in.max() + 1.0])

# Remap with compact CDO grid description file
file_txt = path_work + "grid_target_lonlat.txt"
remap.grid_desc(file_txt, gridtype="lonlat",
                xsize=len(lon_cent), ysize=len(lat_cent),
                xfirst=lon_cent[0], yfirst=lat_cent[0],
                xinc=np.diff(lon_cent).mean(),
                yinc=np.diff(lat_cent).mean())
for i in ("remapbil", "remapcon"):
    cmd = "cdo " + i + "," + file_txt
    sf = file_in
    tf = path_work + sf.split("/")[-1][:-3] + "_lonlat_" + i + ".nc"
    subprocess.call(cmd + " " + sf + " " + tf, shell=True)

# Alternative method: Remap with NetCDF CDO grid description file
# -> this method is useful if (I) CDO does not understand the selected grid
#    description or if (II) user-specific grid modifications are required
lon_cent_2d, lat_cent_2d = np.meshgrid(lon_cent, lat_cent)
lon_edge_2d, lat_edge_2d = np.meshgrid(lon_edge, lat_edge)
file_netcdf = path_work + "grid_target_lonlat.nc"
remap.grid_desc_netcdf(file_netcdf,
                       lon_cent=lon_cent_2d, lat_cent=lat_cent_2d,
                       lon_edge=lon_edge_2d, lat_edge=lat_edge_2d)
for i in ("remapbil", "remapcon"):
    cmd = "cdo " + i + "," + file_netcdf
    sf = file_in
    tf = path_work + sf.split("/")[-1][:-3] + "_lonlat_" + i + "_netcdf.nc"
    subprocess.call(cmd + " " + sf + " " + tf, shell=True)

# Maximal absolute deviation between methods
i_err = "remapcon"  # "remapbil", "remapcon"
ds = xr.open_dataset(path_work + "ERA5-Land_geopotential_lonlat_" + i_err
                     + ".nc")
data_0 = ds["z"].values
ds.close()
ds = xr.open_dataset(path_work + "ERA5-Land_geopotential_lonlat_" + i_err
                     + "_netcdf.nc")
data_1 = ds["z"].values
ds.close()
dev_abs_max = np.abs(data_1 - data_0).max()
print("Maximal absolute deviation: %.10f" % dev_abs_max)

# -----------------------------------------------------------------------------
# Target grid: rotated latitude/longitude grid
# -----------------------------------------------------------------------------

# Define output grid
pole_latitude = 42.5
pole_longitude = -160.0
rlon_cent = np.linspace(-7.5, 7.5, 301)
rlat_cent = np.linspace(-5.0, 5.0, 201)
rlon_edge, rlat_edge = coord_edges(rlon_cent, rlat_cent)
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_latitude,
                                pole_longitude=pole_longitude)

# Check domain coverage
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
lon_frame, lat_frame = grid_frame(lon_edge_in, lat_edge_in, offset=0)
poly = plt.Polygon(list(zip(lon_frame, lat_frame)), facecolor="red",
                   edgecolor="red", alpha=0.2, linewidth=1.0)
ax.add_patch(poly)
rlon_frame, rlat_frame = grid_frame(rlon_edge, rlat_edge, offset=0)
poly = plt.Polygon(list(zip(rlon_frame, rlat_frame)), facecolor="none",
                   edgecolor="blue", alpha=1.0, linewidth=2.5,
                   transform=rot_pole_crs)
ax.add_patch(poly)
ax.coastlines(resolution="50m", color="black", linewidth=1.0)
ax.axis([lon_cent_in[0] - 1.0, lon_cent_in[-1] + 1.0,
         lat_cent_in.min() - 1.0, lat_cent_in.max() + 1.0])

# Remap with compact CDO grid description file
file_txt = path_work + "grid_target_rot.txt"
remap.grid_desc(file_txt, gridtype="projection",
                xsize=len(rlon_cent), ysize=len(rlat_cent),
                xfirst=rlon_cent[0], yfirst=rlat_cent[0],
                xinc=np.diff(rlon_cent).mean(),
                yinc=np.diff(rlat_cent).mean(),
                grid_mapping_name="rotated_latitude_longitude",
                grid_north_pole_longitude=pole_longitude,
                grid_north_pole_latitude=pole_latitude)
for i in ("remapbil", "remapcon"):
    cmd = "cdo " + i + "," + file_txt
    sf = file_in
    tf = path_work + sf.split("/")[-1][:-3] + "_rot_" + i + ".nc"
    subprocess.call(cmd + " " + sf + " " + tf, shell=True)

# -----------------------------------------------------------------------------
# Target grid: map projection
# -----------------------------------------------------------------------------

# Define output grid
lc_crs = ccrs.LambertConformal(central_longitude=20.0, central_latitude=47.5,
                               standard_parallels=(41.5, 53.5))
x_cent = np.arange(-800000.0, 804000.0, 4000.0)
y_cent = np.arange(-400000.0, 404000.0, 4000.0)
x_edge, y_edge = coord_edges(x_cent, y_cent)

# Check domain coverage
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
lon_frame, lat_frame = grid_frame(lon_edge_in, lat_edge_in, offset=0)
poly = plt.Polygon(list(zip(lon_frame, lat_frame)), facecolor="red",
                   edgecolor="red", alpha=0.2, linewidth=1.0)
ax.add_patch(poly)
rlon_frame, rlat_frame = grid_frame(x_edge, y_edge, offset=0)
poly = plt.Polygon(list(zip(rlon_frame, rlat_frame)), facecolor="none",
                   edgecolor="blue", alpha=1.0, linewidth=2.5,
                   transform=lc_crs)
ax.add_patch(poly)
ax.coastlines(resolution="50m", color="black", linewidth=1.0)
ax.axis([lon_cent_in[0] - 1.0, lon_cent_in[-1] + 1.0,
         lat_cent_in.min() - 1.0, lat_cent_in.max() + 1.0])

# Remap with compact CDO grid description file
file_txt = path_work + "grid_target_proj.txt"
remap.grid_desc(file_txt, gridtype="projection",
                xsize=len(x_cent), ysize=len(y_cent),
                xfirst=x_cent[0], yfirst=y_cent[0],
                xinc=np.diff(x_cent).mean(),
                yinc=np.diff(y_cent).mean(),
                grid_mapping_name="lambert_conformal_conic",
                proj_params=lc_crs.proj4_init)
for i in ("remapbil", "remapcon"):
    cmd = "cdo " + i + "," + file_txt
    sf = file_in
    tf = path_work + sf.split("/")[-1][:-3] + "_proj_" + i + ".nc"
    subprocess.call(cmd + " " + sf + " " + tf, shell=True)

###############################################################################
# Remove working directory
###############################################################################

shutil.rmtree(path_work)
