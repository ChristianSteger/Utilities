# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from tqdm import tqdm
import requests


# -----------------------------------------------------------------------------

def download_file(file_url, path_local):
    """Download file from URL and show progress.

    Parameters
    ----------
    file_url : str
        File URL
    path_local : str
        Local path for downloaded file"""

    # Check input arguments
    if not os.path.isdir(path_local):
        raise FileExistsError("Local path does not exist")

    # Download file
    path_local = os.path.join(path_local, "")  # ensure that path ends with "/"
    response = requests.get(file_url, stream=True,
                            headers={"User-Agent": "XY"})
    if response.ok:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 10
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                            unit_scale=True)
        with open(path_local + os.path.split(file_url)[-1], "wb") as infile:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                infile.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ValueError("Inconsistency in file size")
    else:
        raise ValueError("URL response erroneous")
