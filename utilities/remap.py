# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from netCDF4 import Dataset


# -----------------------------------------------------------------------------

def grid_desc(file, gridtype, xsize, ysize, xfirst, yfirst, xinc, yinc,
              **kwargs):
    """Creates a CDO grid description file in which compact grid information
    is saved in a text file.

    Parameters
    ----------
    file : str
        Output name of text file
    gridtype : str
        Type of grid (lonlat or projection)
    xsize : int
        Size of grid in x-direction
    ysize : int
        Size of grid in y-direction
    xfirst : float
        x-coordinate of first grid cell's centre
    yfirst : float
        y-coordinate of first grid cell's centre
    xinc : float
        Grid spacing in x-direction
    yinc : float
        Grid spacing in y-direction
    kwargs : Keyword arguments, optional
        Additional arguments depending on grid

    Returns
    -------
    None

    Notes
    -----
    Source: CDO User Guide (Version 1.9.9) -> 1.4.2.4. CDO grids

    Conversion from float to string with str() -> precision (decimal
    places) are conserved"""

    # Check input arguments
    if gridtype not in ("lonlat", "projection"):
        raise TypeError("invalid value for 'gridtype'")
    if gridtype == "projection":
        if "grid_mapping_name" not in list(kwargs.keys()):
            raise TypeError("no grid mapping name specified")
        if kwargs["grid_mapping_name"] == "rotated_latitude_longitude":
            if (("grid_north_pole_longitude" not in list(kwargs.keys()))
                    or ("grid_north_pole_latitude"
                        not in list(kwargs.keys()))):
                raise TypeError("missing rotated pole information")
        else:
            if "proj_params" not in list(kwargs.keys()):
                raise TypeError("'proj_params' not provided")

    # Regular longitude/latitude grid
    if gridtype == "lonlat":

        file = open(file, "w")
        file.write("#\n")
        file.write("# gridID 1\n")
        file.write("#\n")
        file.write("gridtype  = lonlat\n")
        file.write("xsize     = " + str(xsize) + "\n")
        file.write("ysize     = " + str(ysize) + "\n")
        file.write("xfirst    = " + str(xfirst) + "\n")
        file.write("xinc      = " + str(xinc) + "\n")
        file.write("yfirst    = " + str(yfirst) + "\n")
        file.write("yinc      = " + str(yinc) + "\n")
        file.close()

    # Rotated regular longitude/latitude grid
    elif (gridtype == "projection") \
            and (kwargs["grid_mapping_name"] == "rotated_latitude_longitude"):

        file = open(file, "w")
        file.write("#\n")
        file.write("# gridID 1\n")
        file.write("#\n")
        file.write("gridtype  = projection\n")
        file.write("xsize     = " + str(xsize) + "\n")
        file.write("ysize     = " + str(ysize) + "\n")
        file.write('xunits    = "degrees"\n')
        file.write('yunits    = "degrees"\n')
        file.write("xfirst    = " + str(xfirst) + "\n")
        file.write("xinc      = " + str(xinc) + "\n")
        file.write("yfirst    = " + str(yfirst) + "\n")
        file.write("yinc      = " + str(yinc) + "\n")
        file.write("grid_mapping_name = rotated_latitude_longitude\n")
        file.write("grid_north_pole_longitude = "
                   + str(kwargs["grid_north_pole_longitude"]) + "\n")
        file.write("grid_north_pole_latitude = "
                   + str(kwargs["grid_north_pole_latitude"]) + "\n")
        file.close()

    # Map projections
    else:

        file = open(file, "w")
        file.write("#\n")
        file.write("# gridID 1\n")
        file.write("#\n")
        file.write("gridtype  = projection\n")
        file.write("xsize     = " + str(xsize) + "\n")
        file.write("ysize     = " + str(ysize) + "\n")
        file.write('xunits    = "meter"\n')
        file.write('yunits    = "meter"\n')
        file.write("xfirst    = " + str(xfirst) + "\n")
        file.write("xinc      = " + str(xinc) + "\n")
        file.write("yfirst    = " + str(yfirst) + "\n")
        file.write("yinc      = " + str(yinc) + "\n")
        file.write("grid_mapping = crs\n")
        file.write("grid_mapping_name = " + kwargs["grid_mapping_name"] + "\n")
        file.write('proj_params = "' + kwargs["proj_params"] + '"\n')
        file.close()


# -----------------------------------------------------------------------------

def grid_desc_netcdf(file, lat_cent, lon_cent, lat_edge, lon_edge):
    """Creates a CDO grid description file in which geographic coordinates
    of the grid cell centres and edges are saved in a NetCDF file.

    Parameters
    ----------
    file : str
        Output name of text file
    lat_cent : array_like
        Array (two-dimensional) with latitudes of grid cell centres [deg]
    lon_cent : array_like
        Array (two-dimensional) with longitudes of grid cell centres [deg]
    lat_edge : array_like
        Array (two-dimensional) with latitudes of grid cell edges [deg]
    lon_edge : array_like
        Array (two-dimensional) with longitudes of grid cell edges [deg]

    Returns
    -------
    None

    Notes
    -----
    Source: CDO User Guide (Version 1.9.9) -> 1.4.2.3. SCRIP grids"""

    # Check input arguments
    if ((lat_cent.ndim != 2) or (lon_cent.ndim != 2) or
            (lat_edge.ndim != 2) or (lon_edge.ndim != 2)):
        raise TypeError("Input arrays must be two-dimensional")
    if ((lat_cent.shape != lon_cent.shape) or
            (lat_edge.shape != lon_edge.shape) or
            any([(lat_cent.shape[i] != (lat_edge.shape[i] - 1))
                 for i in range(2)])):
        raise TypeError("Inconsistent shapes of input arrays")

    # Collect grid cell vertices (in clockwise direction)
    lon_edge_ver = np.concatenate((lon_edge[1:, 1:][:, :, np.newaxis],
                                   lon_edge[:-1, 1:][:, :, np.newaxis],
                                   lon_edge[:-1, :-1][:, :, np.newaxis],
                                   lon_edge[1:, :-1][:, :, np.newaxis]),
                                  axis=2)
    lat_edge_ver = np.concatenate((lat_edge[1:, 1:][:, :, np.newaxis],
                                   lat_edge[:-1, 1:][:, :, np.newaxis],
                                   lat_edge[:-1, :-1][:, :, np.newaxis],
                                   lat_edge[1:, :-1][:, :, np.newaxis]),
                                  axis=2)

    # Determine data type of output arrays
    arr_dtype = np.float32
    if any([i.dtype == "float64" for i in (lon_cent, lat_cent,
                                           lon_edge, lat_edge)]):
        arr_dtype = np.float64

    # Write NetCDF file
    ncfile = Dataset(filename=file, mode="w", format="NETCDF4")
    ncfile.createDimension(dimname="grid_rank", size=2)
    nc = ncfile.createVariable(varname="grid_dims", datatype="i",
                               dimensions="grid_rank")
    nc[:] = np.array([lon_cent.shape[1], lon_cent.shape[0]])
    ncfile.createDimension(dimname="grid_size", size=lon_cent.size)
    nc = ncfile.createVariable(varname="grid_center_lat", datatype=arr_dtype,
                               dimensions="grid_size")
    nc.units = "degrees"
    nc.bounds = "grid_corner_lat"
    nc[:] = lat_cent.flatten()
    nc = ncfile.createVariable(varname="grid_center_lon", datatype=arr_dtype,
                               dimensions="grid_size")
    nc.units = "degrees"
    nc.bounds = "grid_corner_lon"
    nc[:] = lon_cent.flatten()
    ncfile.createDimension(dimname="grid_corners", size=4)
    nc = ncfile.createVariable(varname="grid_corner_lat", datatype=arr_dtype,
                               dimensions=("grid_size", "grid_corners"))
    nc.units = "degrees"
    nc[:] = lat_edge_ver.reshape(lon_cent.size, 4)
    nc = ncfile.createVariable(varname="grid_corner_lon", datatype=arr_dtype,
                               dimensions=("grid_size", "grid_corners"))
    nc.units = "degrees"
    nc[:] = lon_edge_ver.reshape(lon_cent.size, 4)
    ncfile.close()
