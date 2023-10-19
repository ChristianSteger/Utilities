# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from utilities.download import download_file

###############################################################################
# Test function 'download_file'
###############################################################################

path_local = "/Users/csteger/Downloads"
file_url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/"\
           + "download/50m/raster/HYP_50M_SR_W.zip"

download_file(file_url, path_local)
