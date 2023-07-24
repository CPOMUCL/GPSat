# read legacy pkl format data to be read in
# expected folder structure for data
# binned observation: <path_to_data_dir>/CS2S3_CPOM/*.pkl
# auxiliary data: <path_to_data_dir>/aux/*.npy
# sea ice extent: <path_to_data_dir>/aux/SIE/*.pkl

# original source of data (as of 2023-02-06) can be found on CPOM server:
# /home/cjn/OI_PolarSnow/EASE/freeboard_daily_processed/CS2S3_CPOM
# auxiliary ("aux") data can be found:
# /home/cjn/OI_PolarSnow/EASE/auxiliary

import numpy as np
import pandas as pd

from GPSat import get_data_path
from GPSat.dataloader import DataLoader

if __name__ == "__main__":

    # ---
    # parameters
    # ---

    # directory containing data - CHANGE AS NEEDED
    data_dir = get_data_path("CS2S3_CPOM")

    # auxiliary data location - CHANGE AS NEEDED
    coord_dir = get_data_path("aux")

    # satellite names - values will be used as prefix
    sats = {
        "CS2": ["CS2_SARIN", "CS2_SAR"],
        "S3A": ["S3A"],
        "S3B": ["S3B"]
    }

    # grid resolution - in km
    grid_res = 25

    # winter season
    season = "2019-2020"
    # season = "2018-2019"

    # store results in h5 file
    output_file = get_data_path("binned", f"cs2s3cpom_{season}_{grid_res}km.h5")

    # --
    # get file names of (binned) data
    # --

    suffix = f"_dailyFB_{grid_res}km_{season}_season.pkl"

    data_files = {k: [f"{l}{suffix}" for l in v] for k, v in sats.items()}
    data_dim_names = ["y", "x", "date"]

    strict = False
    dim_names = data_dim_names

    # ---
    # read in obs data from pkl files, store in Dataset
    # ---

    pkl_df = DataLoader.read_from_pkl_dict(pkl_files=data_files,
                                           pkl_dir=data_dir,
                                           dim_names=data_dim_names,
                                           default_name="obs")

    # ---
    # read in coordinate / auxiliary data
    # ---

    coord_files = {
        "x": f"new_x_{grid_res}km.npy",
        "y": f"new_y_{grid_res}km.npy",
        "lon": f"new_lon_{grid_res}km.npy",
        "lat": f"new_lat_{grid_res}km.npy",
    }
    coord_dims = ["y", "x"]

    coord_arrays = DataLoader.read_from_npy(npy_files=coord_files,
                                            npy_dir=coord_dir,
                                            dims=coord_dims,
                                            return_xarray=False,
                                            flatten_xy=False)

    # convert coord_arrays to DataFrame
    coord_dfs = []
    for k, v in coord_arrays.items():
        midx = pd.MultiIndex.from_product([np.arange(_) for _ in v.shape], names=[f"idx{i}" for i in range(len(v.shape))])
        coord_dfs.append(pd.DataFrame(v.flat, index=midx, columns=[k]))

    coords_df = pd.concat(coord_dfs, axis=1)

    # ---
    # assign coordinates to obs / pkl data
    # ---

    pkl_df = pkl_df.reset_index().merge(coords_df.reset_index(),
                                        on=['idx0', 'idx1'],
                                        how='left')

    pkl_df.drop(["idx0", "idx1"], axis=1, inplace=True)

    pkl_df.rename(columns={"source": "sat"}, inplace=True)

    # ----
    # read SIE data
    # ---

    sie_dir = get_data_path("aux", "SIE")
    sie_file = f"SIE_masking_{grid_res}km_{season}_season.pkl"
    sie_df = DataLoader.read_from_pkl_dict(pkl_files=sie_file,
                                           pkl_dir=sie_dir,
                                           default_name="sie",
                                           dim_names=data_dim_names)

    # sie_ds = sie_ds.assign_coords(coord_arrays)
    sie_df = sie_df.reset_index().merge(coords_df.reset_index(),
                                        on=['idx0', 'idx1'],
                                        how='left')

    sie_df.drop(["idx0", "idx1", "source"], axis=1, inplace=True)

    # ---
    # save data
    # ---

    expert_locs = sie_df
    obs = pkl_df

    # store in hdf5 - NOTE: overwriting file!
    print(f"writing data to:\n{output_file}")
    with pd.HDFStore(output_file, mode="w") as store:
        store.put("data", obs, format='table', data_columns=True)
        store.put("sie", expert_locs, format='table', data_columns=True)
