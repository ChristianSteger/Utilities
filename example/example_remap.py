# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import subprocess
import cartopy.crs as ccrs
import glob
from utilities.grid import coord_edges, grid_frame
from utilities import remap

mpl.style.use("classic")

###############################################################################
# Create working directory
###############################################################################

path_work = os.getenv("HOME") + "/Desktop/work/"
if not os.path.isdir(path_work):
    os.mkdir(path_work)

# Download from https://esgf-data.dkrz.de/search/cordex-dkrz/
# (Project: CORDEX, Product: output, Domain: EUR-11, Experiment: evaluation,
#  Time Frequency: fx, Variable: orog):
# - orog_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-RACMO22E_v1_fx.nc
# - orog_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-ALADIN63_v1_fx.nc
# - orog_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_ICTP-RegCM4-6_v1_fx.nc
# and move files to working directory

###############################################################################
# Source grids other than regular latitude/longitude
###############################################################################

# Define output grid
lon_cent = np.linspace(5.0, 20.0, 301)
lat_cent = np.linspace(42.0, 52.0, 201)
lon_edge, lat_edge = coord_edges(lon_cent, lat_cent)

# Check extent of target domain
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
lon_frame, lat_frame = grid_frame(lon_edge, lat_edge, offset=0)
poly = plt.Polygon(list(zip(lon_frame, lat_frame)), facecolor="none",
                   edgecolor="blue", alpha=1.0, linewidth=2.5)
ax.add_patch(poly)
ax.coastlines(resolution="50m", color="black", linewidth=1.0)
ax.axis([lon_cent[0] - 1.0, lon_cent[-1] + 1.0,
         lat_cent.min() - 1.0, lat_cent.max() + 1.0])

# Create target CDO grid description file
file_txt = path_work + "grid_target_lonlat.txt"
remap.grid_desc(file_txt, gridtype="lonlat",
                xsize=len(lon_cent), ysize=len(lat_cent),
                xfirst=lon_cent[0], yfirst=lat_cent[0],
                xinc=np.diff(lon_cent).mean(),
                yinc=np.diff(lat_cent).mean())

# -----------------------------------------------------------------------------
# Source grid: rotated latitude/longitude grid
# -----------------------------------------------------------------------------

# Get source grid information
file_in = path_work + "orog_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_"\
          + "KNMI-RACMO22E_v1_fx.nc"
ds = xr.open_dataset(file_in)
rlon_cent_in = ds["rlon"].values
rlat_cent_in = ds["rlat"].values
grid_mapping_name = ds["rotated_pole"].grid_mapping_name
pole_longitude = ds["rotated_pole"].grid_north_pole_longitude
pole_latitude = ds["rotated_pole"].grid_north_pole_latitude
ds.close()

# Write grid description file for source grid
file_txt = path_work + "grid_source_rot.txt"
remap.grid_desc(file_txt, gridtype="projection",
                xsize=len(rlon_cent_in), ysize=len(rlat_cent_in),
                xfirst=rlon_cent_in[0], yfirst=rlat_cent_in[0],
                xinc=np.diff(rlon_cent_in).mean(),
                yinc=np.diff(rlat_cent_in).mean(),
                grid_mapping_name="rotated_latitude_longitude",
                grid_north_pole_longitude=pole_longitude,
                grid_north_pole_latitude=pole_latitude)

# Bilinear interpolation
cmd = "cdo remapbil," + path_work + "grid_target_lonlat.txt"
sf = file_in
tf = path_work + sf.split("/")[-1][:-3] + "_lonlat_remapbil.nc"
subprocess.call(cmd + " " + sf + " " + tf, shell=True)

# Conservative interpolation
# -> coordinates of grid cell edges are not provided in NetCDF file of source
#    grid. The above created CDO grid description file is thus used to add
#    this information with 'setgrid'
cmd = "cdo remapcon," + path_work + "grid_target_lonlat.txt"
sf = "-setgrid," + file_txt + " " + file_in
tf = path_work + sf.split("/")[-1][:-3] + "_lonlat_remapcon.nc"
subprocess.call(cmd + " " + sf + " " + tf, shell=True)

# Remove files
files_rm = glob.glob(path_work + "*KNMI-RACMO22E_v1_fx_lonlat*.nc") \
           + [path_work + "grid_source_rot.txt"]
for i in files_rm:
    os.remove(i)

# -----------------------------------------------------------------------------
# Source grid: map projection
# -----------------------------------------------------------------------------

# Conservative interpolation of CNRM-ALADIN63 data
file_in = path_work + "orog_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_"\
          + "CNRM-ALADIN63_v1_fx.nc"
cmd = "cdo remapcon," + path_work + "grid_target_lonlat.txt"
sf = file_in
tf = path_work + sf.split("/")[-1][:-3] + "_lonlat_remapcon.nc"
subprocess.call(cmd + " " + sf + " " + tf, shell=True)
# -> works because grid cell edges are defined

# Conservative interpolation of ICTP-RegCM4-6 data
file_in = path_work + "orog_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_"\
          + "ICTP-RegCM4-6_v1_fx.nc"
ds = xr.open_dataset(file_in)
x_cent_in = ds["x"].values
y_cent_in = ds["y"].values
proj4_params = ds["crs"].proj4_params
ds.close()
# -> requires CDO setgrid command because coordinates of grid cell edges are
#    missing

file_txt = path_work + "grid_source_proj.txt"
remap.grid_desc(file_txt, gridtype="projection",
                xsize=len(x_cent_in), ysize=len(y_cent_in),
                xfirst=x_cent_in[0], yfirst=y_cent_in[0],
                xinc=np.diff(x_cent_in).mean(),
                yinc=np.diff(y_cent_in).mean(),
                grid_mapping_name="lambert_conformal_conic",
                proj_params=proj4_params)

cmd = "cdo remapcon," + path_work + "grid_target_lonlat.txt"
sf = "-setgrid," + file_txt + " " + file_in
tf = path_work + sf.split("/")[-1][:-3] + "_lonlat_remapcon.nc"
subprocess.call(cmd + " " + sf + " " + tf, shell=True)

# Remove files
files_rm = glob.glob(path_work + "*lonlat_remapcon.nc") \
           + [path_work + "grid_source_proj.txt"]
for i in files_rm:
    os.remove(i)

# Notes
# NetCDF file with remapping weights can be generated if a lot of data
# on the same grid has to be remapped:
# cdo gencon,grid_target.txt input.nc weights.nc
# cdo remap,grid_target.txt,weights.nc input.nc output.nc
