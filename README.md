# Utilities
Various utilities for processing climate model data with Python.

# Package dependencies

The following Python packages are required to run Utilities: NumPy, Shapely, Matplotlib, netCDF4.
Running the tests requires the additional packages: Xarray, descartes, Fiona, pyproj, cartopy.

# Installation

First, create a Conda environment with all the required Python packages:

```bash
conda create -n utilities -c conda-forge numpy matplotlib netcdf4 shapely xarray pyproj cartopy descartes fiona
```

and **activate this environment**. The Utilities package can then be installed with:

```bash
git clone https://github.com/ChristianSteger/Utilities.git
cd Utilities
python -m pip install .
```

# Grid
Functions related to regular climate model grids:
- **coord_edges()**: Compute edge coordinates from grid cell centre coordinates. Only works for grid with regular spacing.
- **grid_frame()**: Compute frame around a grid with a certain offset from the outer boundary.
- **area_gridcells()**: Compute area of grid cells. Assume plane (Euclidean) geometry.
- **polygon_inters_exact()**: "Compute area fractions of grid cells located inside the polygon. Exact method in which individual grid cells are intersected with polygon. Assume plane (Euclidean) geometry.
- **area_gridcells()**: Compute area fractions of grid cells located inside the polygon. Approximate method in which intersecting areas are derived by checking points within the grid cells (one ore multiple sampling). Assume plane (Euclidean) geometry.

# Plot
Functions related to plotting with Matplotlib:
- **truncate_colormap()**: Truncate colormap to specific range between [0.0, 1.0].

# Remap
Functions related to remapping gridded data with [CDO](https://code.mpimet.mpg.de/projects/cdo/): 
- **grid_desc()**: Creates a CDO grid description file in which compact grid information is saved in a text file.
- **grid_desc_netcdf()**: Creates a CDO grid description file in which geographic coordinates of the grid cell centres and edges are saved in a NetCDF file.

# Miscellaneous
Miscellaneous functions:
- **aggregation_1d()**: Aggregate one-dimensional array.
- **aggregation_2d()**: Aggregate two-dimensional array.
- **nanaverage()**: Compute weighted average from non-NaN-values.
- **bool_mask_extend()**: Extend *True* region in two-dimensional boolean mask by one grid cell in every of the eight directions.

# Support and collaboration

In case of issues or questions, contact Christian R. Steger (christian.steger@env.ethz.ch). Please report any bugs you find in Utilities. You are welcome to fork this repository to modify the source code.