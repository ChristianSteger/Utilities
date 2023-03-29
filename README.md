# Utilities
Various utilities for processing climate model data with Python

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
List of available functions:
- **coord_edges()**: Compute edge coordinates from grid cell centre coordinates. Only works for grid with regular spacing.
- **grid_frame()**: Compute frame around a grid with a certain offset from the outer boundary.
- **area_gridcells()**: Compute area of grid cells. Assume plane (Euclidean) geometry.
- **polygon_inters_exact()**: "Compute area fractions of grid cells located inside the polygon. Exact method in which individual grid cells are intersected with polygon. Assume plane (Euclidean) geometry.
- **area_gridcells()**: Compute area fractions of grid cells located inside the polygon. Approximate method in which intersecting areas are derived by checking points within the grid cells (one ore multiple sampling). Assume plane (Euclidean) geometry.

# Plot

# Remap

# Miscellaneous

# Support and collaboration

In case of issues or questions, contact Christian R. Steger (christian.steger@env.ethz.ch). Please report any bugs you find in Utilities. You are welcome to fork this repository to modify the source code.