# Utilities
Various utilities for processing and plotting climate model data with Python. *Grid* and *Remap* functions are designed for model output on a regular horizontal grid (rotated latitude/longitude or map projection).

# Package dependencies

The following Python packages are required to run Utilities: NumPy, Matplotlib, Shapely and netcdf4.
Running the example and the tests requires the additional packages Xarray, pyproj, cartopy and Fiona.
The *Remap* functions require [CDO](https://code.mpimet.mpg.de/projects/cdo/) to be installed and to be available in the terminal.

# Installation

First, ensure that all required Python packages are available in the environment you want to install Utilities.
The essential packages are installed with

```bash
conda install -c conda-forge numpy matplotlib shapely netcdf4 requests tqdm
```

and the optional ones with

```bash
conda install -c conda-forge xarray pyproj cartopy fiona
```

The Utilities package can then be installed with:

```bash
git clone https://github.com/ChristianSteger/Utilities.git
cd Utilities
python -m pip install .
```

# Functions

## Download
Functions related to downloading files from the Web:
- **download_file()**: Download file from URL and show progress.
- **unzip_file()**: Unzip file.

## Grid
Functions related to regular climate model grids:
- **coord_edges()**: Compute edge coordinates from grid cell centre coordinates. Only works for grid with regular spacing.
- **grid_frame()**: Compute frame around a grid with a certain offset from the outer boundary.
- **area_gridcells()**: Compute area of grid cells. Assume plane (Euclidean) geometry.
- **polygon_inters_exact()**: "Compute area fractions of grid cells located inside a polygon. Exact method in which individual grid cells are intersected with the polygon. Assume plane (Euclidean) geometry.
- **polygon_inters_approx()**: Compute area fractions of grid cells located inside a polygon. Approximate method in which intersecting areas are derived by checking points within the grid cells (single or multiple sampling). Assume plane (Euclidean) geometry.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Utilities/Grid_polygon_inters.png?raw=true "Output from test_grid.py")
- **polygon_rectangular()**: Create rectangular shapely polygon with specified verticies spacing.

## Plot
Functions related to plotting with Matplotlib:
- **truncate_colormap()**: Truncate colormap to specific range between [0.0, 1.0].
- **polygon2patch()**: Convert Shapely (multi-)polygon to Matplotlib PatchCollection.

## Remap
Functions related to remapping gridded data with [CDO](https://code.mpimet.mpg.de/projects/cdo/): 
- **grid_desc()**: Creates a CDO grid description file in which compact grid information is saved in a text file.
- **grid_desc_netcdf()**: Creates a CDO grid description file in which geographic coordinates of the grid cell centres and edges are saved in a NetCDF file.

## Miscellaneous
Miscellaneous functions:
- **aggregation_1d()**: Aggregate one-dimensional array.
- **aggregation_2d()**: Aggregate two-dimensional array.
- **nanaverage()**: Compute weighted average from non-NaN-values.
- **bool_mask_extend()**: Extend *True* region in two-dimensional boolean mask by one grid cell in every of the eight directions.
- **consecutive_length_max()**: Compute maximal length of consecutive *True* values along the first dimension of a three-dimensional array. Optionally return the range (start and stop) indices of this sequence. In case sequence with the maximal length occurs multiple times, the indices of the first sequence is returned.

# Examples
- **example_remap.py**: Examples for remapping non-regular latitude/longitude to regular latitude/longitude grids with CDO. Requires some example [EURO-CORDEX](https://esgf-data.dkrz.de/search/cordex-dkrz/) NetCDF files.

# Support and collaboration

In case of issues or questions, contact Christian R. Steger (christian.steger@env.ethz.ch). Please report any bugs you find in Utilities. You are welcome to fork this repository to modify the source code.