# read legacy pkl format data to be read in
# expected folder structure for data
# binned observation: <path_to_data_dir>/CS2S3_CPOM/*.pkl
# auxiliary data: <path_to_data_dir>/aux/*.npy
# sea ice extent: <path_to_data_dir>/aux/SIE/*.pkl

import xarray as xr

from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.dataloader import DataLoader

if __name__ == "__main__":

    # ---
    # parameters
    # ---

    # --
    # (binned) observation data
    # --

    sats = {
        "CS2": ["CS2_SARIN", "CS2_SAR"],
        "S3A": ["S3A"],
        "S3B": ["S3B"]
    }
    grid_res = 25
    # season = "2019-2020"
    season = "2018-2019"

    data_dir = get_data_path("CS2S3_CPOM")
    suffix = f"_dailyFB_{grid_res}km_{season}_season.pkl"

    data_files = {k: [f"{l}{suffix}" for l in v] for k, v in sats.items()}
    data_dim_names = ["y", "x", "date"]

    strict = False
    dim_names = data_dim_names

    # ---
    # read in obs data from pkl files, store in Dataset
    # ---

    pkl_ds = DataLoader.read_from_pkl_dict(pkl_files=data_files,
                                           pkl_dir=data_dir,
                                           dim_names=data_dim_names)

    # ---
    # read in coordinate / auxiliary data
    # ---

    coord_dir = get_data_path("aux")
    coord_files = {
        "x": f"new_x_{grid_res}km.npy",
        "y": f"new_y_{grid_res}km.npy",
        "lon": f"new_lon_{grid_res}km.npy",
        "lat": f"new_lat_{grid_res}km.npy",
    }
    coord_dims = ["y", "x"]

    coord_arrays = DataLoader.read_from_npy(npy_files=coord_files,
                                            npy_dir=coord_dir,
                                            dims=coord_dims)

    # ---
    # assign coordinates to obs / pkl data
    # ---

    # TODO: review this, see if it can be done a bit more cleanly
    names = list(pkl_ds.keys())
    _ = [pkl_ds[_] for _ in names]
    da = xr.concat(_, dim=xr.DataArray(names, dims=['sat']))
    da.name = "obs"

    da = da.assign_coords(coord_arrays)

    # ----
    # read SIE data
    # ---

    sie_dir = get_data_path("aux", "SIE")
    sie_files = f"SIE_masking_{grid_res}km_{season}_season.pkl"
    sie_ds = DataLoader.read_from_pkl_dict(pkl_files=sie_files,
                                           pkl_dir=sie_dir,
                                           default_name="sie",
                                           dim_names=data_dim_names)

    sie_ds = sie_ds.assign_coords(coord_arrays)



