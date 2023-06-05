#%%
import json
import re

import pandas as pd
import numpy as np
import numba as nb

from typing import List, Dict, Union
from dataclasses import dataclass
from PyOptimalInterpolation.local_experts import get_results_from_h5file
from PyOptimalInterpolation.utils import json_serializable, cprint
from PyOptimalInterpolation import get_data_path, get_parent_path
from PyOptimalInterpolation.models import get_model


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
                 (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
                  nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:])],
                '(), (), (n), (n), (), (), (n) -> ()',
                nopython=True, target='parallel')
def gaussian_2d_weight(x0, y0, x, y, l_x, l_y, vals, out):
    """weight functions of the form exp(-d^2), where d is the distance between reference position
    (x0, y0) and the others"""

    # calculate the squared distance from the reference equation (normalising dist in each dimension by a length_scale)
    # - can they be specified with defaults?
    d2 = ((x-x0)/l_x[0]) ** 2 + ((y - y0)/l_y[0])**2

    # get the weight function (un-normalised)
    w = np.exp(-d2/2)

    # get the weighted sum of vals, skipping vals which are nan
    w_sum = 0
    w_val = 0
    for i in range(len(vals)):
        if ~np.isnan(vals[i]):
            w_val += w[i] * vals[i]
            w_sum += w[i]

    # if all weights are zero, i.e. in the case all nan vals, return np.nan
    if w_sum == 0:
        out[0] = np.nan
    # otherwise return the normalised weighted value
    else:
        out[0] = w_val / w_sum


@dataclass
class SmoothingConfig:
    l_x: Union[int, float] = 1
    l_y: Union[int, float] = 1
    max: Union[int, float] = None
    min: Union[int, float] = None


def smooth_hyperparameters(result_file: str,
                           params_to_smooth: List[str],
                           smooth_config_dict: Dict[str, SmoothingConfig],
                           xy_dims: List[str] = ['x', 'y'],
                           reference_table_suffix: str = "",
                           table_suffix: str = "_SMOOTHED",
                           output_file: str = None,
                           save_config_file: bool = True):
    
    assert table_suffix != reference_table_suffix
    assert all(param in smooth_config_dict.keys() for param in params_to_smooth)

    # extract the dimensions to smooth over, will be used to make a 2d array
    assert len(xy_dims) == 2, "dimensions to smooth over must have length 2"
    x_col, y_col = xy_dims  

    # Get model
    with pd.HDFStore(result_file, mode="r") as store:
        run_details = store.select("run_details")
    model_name = run_details['model'].iloc[0]
    # Extract model name which comes after the last "."
    match = re.search(r'\.(\w+)$', model_name)
    if match:
        model_name = match.group(1)

    model = get_model(model_name)

    # Instantiate model with pseudo data
    data = [0., 1.]
    columns = ['x', 'y']
    data = pd.DataFrame([data], columns=columns)
    coords_col = 'x'
    obs_col = 'y'

    model_ = model(data, coords_col=coords_col, obs_col=obs_col)
    all_params = model_.param_names

    other_params = [x for x in all_params if x not in params_to_smooth]

    smooth_params_with_suffix = [f"{param}{reference_table_suffix}" for param in params_to_smooth]
    other_params_with_suffix = [f"{param}{reference_table_suffix}" for param in other_params]
    smooth_config_dict = {f"{k}{reference_table_suffix}": v for k, v in smooth_config_dict.items()}

    # ----
    # read in all hyper parameters
    # ----
    select_tables = all_params + ["expert_locs", "oi_config"]
    dfs, oi_configs = get_results_from_h5file(result_file,
                                              global_col_funcs=None,
                                              merge_on_expert_locations=False,
                                              select_tables=select_tables,
                                              table_suffix=reference_table_suffix,
                                              add_suffix_to_table=True)
    
    coords_col = oi_configs[-1]['data']['coords_col']

    # -----
    # smooth-out hyper parameter fields
    # -----

    out = {}

    for hp_idx, hp in enumerate(smooth_params_with_suffix):
        # if current hyper parameter is specified in the smooth dict
        if hp in smooth_config_dict:
            df = dfs[hp].copy(True)

        df_org_col_order = df.columns.values.tolist()

        # select hyperparameter field to smooth out
        smooth_config = smooth_config_dict[hp]
        # get the other (None smoothing) dimensions, to iterate over
        other_dims = [c for c in coords_col if c not in xy_dims]
        # add the other "_dim_*" columns
        dim_cols = [c for c in df.columns if re.search("^_dim_\d", c)]
        other_dims += dim_cols
        # get the unique combinations of other_dims, used to select subset of data
        unique_odims = df[other_dims].drop_duplicates()

        # increment over the rows -want to get a DataFrame representation of each row
        smooth_list = []
        for idx, row in unique_odims.iterrows():
            # get the row as a DataFrame
            row_df = row.to_frame().T

            # and merge on the other dim columns. This extracts a subset of df for the given row value.
            row_df = row_df.merge(df,
                                  on=other_dims,
                                  how='inner')
            
            val_col = params_to_smooth[hp_idx]

            x0, y0 = [row_df[c].values for c in xy_dims]
            x, y = [row_df[c].values for c in xy_dims]
            vals = row_df[val_col].values

            if smooth_config.max is not None:
                vals[vals > smooth_config.max] = smooth_config.max

            if smooth_config.min is not None:
                vals[vals < smooth_config.min] = smooth_config.min

            l_x, l_y = smooth_config.l_x, smooth_config.l_y

            smoothed_hyperparameter_field = gaussian_2d_weight(x0, y0, x, y, l_x, l_y, vals)
            row_df[val_col] = smoothed_hyperparameter_field

            # create a new tmp dataframe with just val, x, y cols - other dims to be added
            tmp = row_df[[val_col, x_col, y_col]].copy(True)

            # drop nans
            tmp.dropna(inplace=True)

            # add in the 'other dimension' values
            for od in other_dims:
                tmp[od] = row[od]

            # re-order columns to previous order: strictly not needed
            tmp = tmp[df_org_col_order]

            smooth_list.append(tmp)

        smooth_df = pd.concat(smooth_list)
        # set index to be coordinates column
        smooth_df.set_index(coords_col, inplace=True)

        out_table = f'{hp}{table_suffix}'
        cprint(f"adding smoothed table: {out_table}", c="OKCYAN")
        out[out_table] = smooth_df

        # being lazy: make a copy of smooth_config used for this out_table
        smooth_config_dict[out_table] = smooth_config

    # ---
    # copy non-smoothed hyper parameters to table
    # ---
    for param in other_params_with_suffix:
        out_table = f'{param}{table_suffix}'
        try:
            cprint(f"copying table: {param} to {out_table}", c="OKCYAN")
            out[out_table] = dfs[param].copy(True)
            out[out_table].set_index(coords_col, inplace=True)
        except KeyError as e:
            cprint(f"{e} not found, skipping", c="FAIL")

    # ---
    # write results to table
    # ---
    output_file = result_file if output_file is None else output_file
    cprint(f"writing (smoothed) hyper parameters to:\n{output_file}\ntable_suffix:_SMOOTHED", c="OKGREEN")
    with pd.HDFStore(output_file, mode="a") as store:
        for k, v in out.items():
            # out_table = f"{k}{table_suffix}"
            cprint(f"writing: {k} to table", c="BOLD")
            # TODO: confirm this will overwrite existing table?
            store.put(k, v, format="table", append=False)

            store_attrs = store.get_storer(k).attrs
            try:
                store_attrs['smooth_config'] = smooth_config_dict[k]
            except KeyError as e:
                org_table = re.sub(f"{table_suffix}$", "", k)
                store_attrs['smooth_config'] = {"comment": f"no smoothing, copied directly from {org_table}"}

    # ---
    # write the configs to file (optional). Maybe output LocalExpertConfig dataclass?
    # ---
    if save_config_file:
        new_pred_loc = None # what is this?
        out_config = re.sub("\.h5$", f"{reference_table_suffix}{table_suffix}.json", result_file)
        tmp = []
        for oic in oi_configs:
            # change, update the run kwargs to not optimise and use the table_suffix
            run_kwargs = oic.get("run_kwargs", {})
            run_kwargs["optimise"] = False
            run_kwargs["table_suffix"] = f"{reference_table_suffix}{table_suffix}"
            run_kwargs["store_path"] = output_file

            # add load_params - load from self
            model = oic["model"]
            model["load_params"] = {
                "file": output_file,
                "table_suffix": f"{reference_table_suffix}{table_suffix}"
            }

            oic["run_kwargs"] = run_kwargs
            oic["model"] = model

            if new_pred_loc is not None:
                oic["pred_loc"] = new_pred_loc

            tmp.append(json_serializable(oic))

        cprint(f"writing config (to use to make predictions with smoothed values) to:\n{out_config}", c="OKBLUE")
        with open(out_config, "w") as f:
            json.dump(tmp, f, indent=4)

#%%
if __name__ == "__main__":
    # from PyOptimalInterpolation import get_data_path, get_parent_path
    # from PyOptimalInterpolation.models import get_model

    # result_file=get_parent_path("results", "example", "ABC_50km_test.h5")
    # with pd.HDFStore(result_file, mode="r") as store:
    #     run_deets = store.select("run_details")

    # model_name = run_deets['model'].iloc[0]
    # match = re.search(r'\.(\w+)$', model_name)

    # if match:
    #     model_name = match.group(1)
    #     print(model_name)

    # model = get_model(model_name)

    # # Instantiate model with pseudo data
    # data = [0., 1.]
    # columns = ['x', 'y']
    # data = pd.DataFrame([data], columns=columns)
    # coords_col = 'x'
    # obs_col = 'y'

    # model_instance = model(data, coords_col=coords_col, obs_col=obs_col)

    # print(model_instance.param_names)

    out_file = get_parent_path("results", "example", "ABC_50km_test_SMOOTHED.h5") # Path to store smoothed hyperparameters
    smooth_configs = {"lengthscales": SmoothingConfig(l_x=200_000, l_y=200_000, max=12),
                    "likelihood_variance": SmoothingConfig(l_x=200_000, l_y=200_000),
                    "kernel_variance": SmoothingConfig(l_x=200_000, l_y=200_000, max=0.1)}

    smooth_hyperparameters(result_file=get_parent_path("results", "example", "ABC_50km_test.h5"),
                        params_to_smooth=["lengthscales", "likelihood_variance", "kernel_variance"],
                        smooth_config_dict=smooth_configs,
                        output_file=out_file,
                        save_config_file=False)


# %%
