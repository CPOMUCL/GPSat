# Modules for postprocessing data after training / inference

import json
import re
import os
import warnings

import pandas as pd
import numpy as np
import numba as nb

from typing import List, Dict, Union
from dataclasses import dataclass
from scipy.stats import norm
from GPSat.local_experts import get_results_from_h5file
from GPSat.utils import json_serializable, cprint, get_config_from_sysargv, nested_dict_literal_eval
from GPSat import get_data_path, get_parent_path
from GPSat.models import get_model



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
    """
    Configuration used for hyperparameter smoothing.

    Attributes
    ----------
    l_x: int or float, default 1
        The lengthscale (x-direction) parameter for Gaussian smoothing.
    l_y: int or float, default 1
        The lengthscale (y-direction) parameter for Gaussian smoothing.
    max: int or float, optional
        Maximal value that the hyperparameter can take.
    min: int or float, optional
        Minimal value that the hyperparameter can take.

    Notes
    -----
    This configuration is used to smooth 2D hyperparameter fields.

    """
    l_x: Union[int, float] = 1
    l_y: Union[int, float] = 1
    max: Union[int, float] = None
    min: Union[int, float] = None

    def __getitem__(self, item):
        # magic method for indexing: e.g.  SmoothingConfig()['l_x']
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(f"{item} is not an attribute of {type(self)} object")

    def get(self, key, default=None):
        # method to replicate dict.get, however it won't have the exact same behaviour
        if hasattr(self, key):
            return self[key]
        else:
            return default


def smooth_hyperparameters(result_file: str,
                           params_to_smooth: List[str],
                           smooth_config_dict: Dict[str, SmoothingConfig],
                           # l_x: Union[Union[int, float], List[Union[int, float]]] = 1,
                           # l_y: Union[Union[int, float], List[Union[int, float]]] = 1,
                           # max: Union[None, Union[int, float], List[Union[int, float]]] = None,
                           # min: Union[None, Union[int, float], List[Union[int, float]]] = None,
                           xy_dims: List[str] = ['x', 'y'],
                           reference_table_suffix: str = "",
                           table_suffix: str = "_SMOOTHED",
                           output_file: str = None,
                           model_name: str = None,
                           save_config_file: bool = True):
    """
    Smooth hyperparameters in an HDF5 results file using Gaussian smoothing.

    Parameters
    ----------
    result_file: str
        The path to the HDF5 results file.
    params_to_smooth: list of str
        A list of hyperparameters to be smoothed.
    smooth_config_dict: Dict[str, SmoothingConfig]
        A dictionary specifying smoothing configurations for each hyperparameter. This should be a dictionary \
        where keys are hyperparameter names, and values are instances of the \
        :class:`SmoothingConfig <GPSat.postprocessing.SmoothingConfig>` class specifying smoothing parameters.
    xy_dims: list of str, default ['x', 'y']
        The dimensions to use for smoothing (default: ``['x', 'y']``).
    reference_table_suffix: str, default ""
        The suffix to use for reference table names (default: ``""``).
    table_suffix: str, default "_SMOOTHED"
        The suffix to add to smoothed hyperparameter table names (default: ``"_SMOOTHED"``).
    output_file: str, optional
        The path to the output HDF5 file to store smoothed hyperparameters.
    model_name: str, optional
        The name of the model for which hyperparameters are being smoothed.
    save_config_file: bool, optional
        Whether to save a configuration file for making predictions with smoothed values.

    Returns
    -------
    None

    Notes
    -----
    - This function applies Gaussian smoothing to specified hyperparameters in an HDF5 results file.
    - The ``output_file`` parameter allows you to specify a different output file for storing the smoothed hyperparameters.
    - If ``model_name`` is not provided, it will be determined from the input HDF5 file.
    - If ``save_config_file`` is ``True``, a configuration file for making predictions with smoothed values will be saved.

    """
    
    assert table_suffix != reference_table_suffix

    # REMOVED below, revert back to smooth_config_dict being required input
    # as newer implementation is not backwards compatible with default/example configuration
    # DONT USE max, min AS VARIABLE NAMES AS THEY PYTHON FUNCTIONS

    # Set up config class to pass through Gaussian smoothing module
    # if isinstance(l_x, (int, float)):
    #     l_x = [l_x for _ in params_to_smooth]
    # if isinstance(l_y, (int, float)):
    #     l_y = [l_y for _ in params_to_smooth]
    # if isinstance(max, (int, float)) or max is None:
    #     max = [max for _ in params_to_smooth]
    # if isinstance(l_y, (int, float)) or min is None:
    #     min = [min for _ in params_to_smooth]

    # assert len(l_x) == len(params_to_smooth)
    # assert len(l_y) == len(params_to_smooth)
    # assert len(max) == len(params_to_smooth)
    # assert len(min) == len(params_to_smooth)

    # smooth_config_dict = {}
    # for i, param in enumerate(params_to_smooth):
    #
    #     smooth_config_dict[param] = SmoothingConfig(l_x[i], l_y[i], max[i], min[i])

    # tmp = {}
    # for k, v in smooth_config_dict.items():
    #     tmp[k] = SmoothingConfig(**v)
    # smooth_config_dict = tmp

    # extract the dimensions to smooth over, will be used to make a 2d array
    assert len(xy_dims) == 2, "dimensions to smooth over must have length 2"
    x_col, y_col = xy_dims  

    # if model name is not specified, get from run_details
    if model_name is None:
        # Get model and retrieve parameter names (TODO: A bit hacky. Better way to do this?)
        with pd.HDFStore(result_file, mode="r") as store:
            run_details = store.select(f"run_details{reference_table_suffix}")

        unique_models = run_details['model'].unique()
        assert len(unique_models) == 1, f"more than one model was found in run_details{reference_table_suffix}, not sure which to use"
        model_name = unique_models[-1]
        # Extract model name which comes after the last "."
        model_name = model_name.split(".")[-1]

        print(f"found model_name: {model_name}")
    else:
        print(f"provided model_name: {model_name}")

    # Instantiate model with pseudo data - only used to get param_names from model
    model = get_model(model_name)

    data = [0., 1.]
    columns = ['x', 'y']
    data = pd.DataFrame([data], columns=columns)
    coords_col = 'x'
    obs_col = 'y'

    model_ = model(data, coords_col=coords_col, obs_col=obs_col)

    # Extract parameter names from model
    all_params = model_.param_names
    assert all([pts in all_params for pts in params_to_smooth ]), \
        f"some params_to_smooth:\n{params_to_smooth} are not in model.param_names:\n{all_params}"

    # other parameters will be copied
    other_params = [x for x in all_params if x not in params_to_smooth]

    smooth_params_with_suffix = [f"{param}{reference_table_suffix}" for param in params_to_smooth]
    other_params_with_suffix = [f"{param}{reference_table_suffix}" for param in other_params]
    smooth_config_dict = {f"{k}{reference_table_suffix}": v for k, v in smooth_config_dict.items()}

    # ----
    # read in all hyper parameters
    # ----
    select_tables = all_params # + [f"expert_locs{reference_table_suffix}", f"oi_config{reference_table_suffix}"]
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
        else:
            raise NotImplementedError(f"parameter: {hp} is not in dfs: {list(dfs.keys())}\ncheck your inputs")

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

            if smooth_config.get("max", None) is not None:
                vals[vals > smooth_config["max"]] = smooth_config["max"]

            if smooth_config.get("min", None) is not None:
                vals[vals < smooth_config["min"]] = smooth_config["min"]

            # l_x, l_y = smooth_config.l_x, smooth_config.l_y
            l_x, l_y = smooth_config["l_x"], smooth_config["l_y"]

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
    cprint(f"writing (smoothed) hyper parameters to:\n{output_file}\ntable_suffix:{table_suffix}", c="OKGREEN")
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
    # write the configs to file (optional). Maybe output ExperimentConfig dataclass?
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

        return out_config

def glue_local_predictions(preds_df: pd.DataFrame,
                           inference_radius: pd.DataFrame,
                           R: Union[int, float, list]=3
                           ) -> pd.DataFrame:
    """
    DEPRECATED. See ``glue_local_predictions_1d`` and ``glue_local_predictions_2d``.
    Glues overlapping predictions by taking a normalised Gaussian weighted average.

    WARNING: This method only deals with expert locations on a regular grid

    Parameters
    ----------
    preds_df: pd.DataFrame
        containing predictions generated from local expert OI. It should have the following columns:
        - pred_loc_x (float): The x-coordinate of the prediction location.
        - pred_loc_y (float): The y-coordinate of the prediction location.
        - f* (float): The predictive mean at the location (pred_loc_x, pred_loc_y).
        - f*_var (float): The predictive variance at the location (pred_loc_x, pred_loc_y).
    expert_locs_df: pd.DataFrame
        containing local expert locations used to perform OI. It should have the following columns:
        - x (float): The x-coordinate of the expert location.
        - y (float): The y-coordinate of the expert location.
    sigma: int, float, or list, default 3
        The standard deviation of the Gaussian weighting in the x and y directions.
        If a single value is provided, it is used for both directions.
        If a list is provided, the first value is used for the x direction and the second value is used for the y direction. Defaults to 3.

    Returns
    -------
    pd.DataFrame:
        dataframe consisting of glued predictions (mean and std). It has the following columns:
        - pred_loc_x (float): The x-coordinate of the prediction location.
        - pred_loc_y (float): The y-coordinate of the prediction location.
        - f* (float): The glued predictive mean at the location (pred_loc_x, pred_loc_y).
        - f*_std (float): The glued predictive standard deviation at the location (pred_loc_x, pred_loc_y).

    Notes
    -----
    The function assumes that the expert locations are equally spaced in both the x and y directions.
    The function uses the scipy.stats.norm.pdf function to compute the Gaussian weights.
    The function normalizes the weighted sums with the total weights at each location.


    """

    # TODO: confirm notes in docstring are accurate
    preds = preds_df.copy(deep=True)
    # Compute Gaussian weights
    preds['weights_x'] = norm.pdf(preds['pred_loc_x'], preds['x'], inference_radius/R)
    preds['weights_y'] = norm.pdf(preds['pred_loc_y'], preds['y'], inference_radius/R)
    preds['total_weights'] = preds['weights_x'] * preds['weights_y']
    # Multiply predictive mean and std by weights
    preds['f*'] = preds['f*'] * preds['total_weights']
    preds['f*_var'] = preds['f*_var'] * preds['total_weights']
    preds['y_var'] = preds['y_var'] * preds['total_weights']
    # Compute weighted sum of mean and std, in addition to the total weights at each location
    glued_preds = preds[['pred_loc_x', 'pred_loc_y',  'total_weights', 'f*', 'f*_var', 'y_var']].groupby(['pred_loc_x', 'pred_loc_y']).sum()
    glued_preds = glued_preds.reset_index()
    # Normalise weighted sums with total weights
    glued_preds['f*'] = glued_preds['f*'] / glued_preds['total_weights']
    glued_preds['f*_var'] = glued_preds['f*_var'] / glued_preds['total_weights']
    glued_preds['y_var'] = glued_preds['y_var'] / glued_preds['total_weights']
    return glued_preds.drop("total_weights", axis=1)


def glue_local_predictions_1d(preds_df: pd.DataFrame,
                              pred_loc_col: str,
                              xprt_loc_col: str,
                              vars_to_glue: Union[str, List[str]],
                              inference_radius: Union[int, float, dict],
                              R=3
                              ) -> pd.DataFrame:
    """
    Glues together overlapping local expert predictions in 1D by Gaussian-weighted averaging.

    Parameters
    ----------
    preds_df: pandas dataframe
        A dataframe containing the results of local experts predictions.
        The dataframe should have columns containing the (1) prediction locations,
        (2) expert locations, and (3) any predicted variables we wish to glue (e.g. the predictive mean).
    pred_loc_col: str
        The column in the results dataframe corresponding to the prediction locations
    xprt_loc_col: str
        The column in the results dataframe corresponding to the local expert locations
    vars_to_glue: str | list of strs
        The column(s) corresponding to variables we wish to glue (e.g. the predictive mean and variance).
    inference_radius: int | float | dict
        The inference radius for each local experts. If specified as a dict, the keys should be the 
        expert locations and the corresponding values should be the corresponding inference radius of that expert.
        If specified as an int or float, it assumes that all experts have the same inference radius.
    R: int | float, default 3
        A weight controlling the standard deviation of the Gaussian weights. The standard deviation will be given by
        the formula ``std = inference_radius / R``. The default value of 3 will place 99% of the Gaussian mass
        within the inference radius.

    Returns
    -------
    pandas dataframe
        A dataframe of glued predictions, whose columns contain (1) the prediction locations and (2) the glued variables.

    """
                              
    if isinstance(vars_to_glue, str):
        vars_to_glue = [vars_to_glue]

    preds = preds_df.copy(deep=True)
    
    if isinstance(inference_radius, dict):
        assert len(inference_radius) == len(preds[xprt_loc_col].unique()), print("...")
        inference_radius_ = [inference_radius[loc] for loc in preds[xprt_loc_col]]
        inference_radius = np.array(inference_radius_)
    elif isinstance(inference_radius, (int, float)):
        pass
    else:
        print(f"inference_radius must be int, float or dict.")

    # Compute Gaussian weights
    preds['weights'] = norm.pdf(preds[pred_loc_col], preds[xprt_loc_col], inference_radius/R)
    preds['total_weights'] = preds['weights']

    # Multiply variables we wish to glue by Gaussian weights
    for var in vars_to_glue:
        preds[var] = preds[var] * preds['weights']

    # Compute weighted sum of variables, in addition to the total weights at each location
    glued_preds = preds[[pred_loc_col,  'total_weights'] + vars_to_glue].groupby([pred_loc_col]).sum()
    glued_preds = glued_preds.reset_index()
    
    # Normalise weighted sums with total weights
    for var in vars_to_glue:
        glued_preds[var] = glued_preds[var] / glued_preds['total_weights']
        
    return glued_preds.drop("total_weights", axis=1)


def glue_local_predictions_2d(preds_df: pd.DataFrame,
                              pred_loc_cols: List[str],
                              xprt_loc_cols: List[str],
                              vars_to_glue: Union[str, List[str]],
                              inference_radius: Union[int, float, dict],
                              R=3
                              ) -> pd.DataFrame:
    """
    Glues together overlapping local expert predictions in 2D by Gaussian-weighted averaging.

    Parameters
    ----------
    preds_df: pandas dataframe
        A dataframe containing the results of local experts predictions.
        The dataframe should have columns containing the (1) prediction locations,
        (2) expert locations, and (3) any predicted variables we wish to glue (e.g. the predictive mean).
    pred_loc_col: list of strs
        The xy-columns in the results dataframe corresponding to the prediction locations
    xprt_loc_cols: list of strs
        The xy-columns in the results dataframe corresponding to the local expert locations
    vars_to_glue: str | list of strs
        The column(s) corresponding to variables we wish to glue (e.g. the predictive mean and variance).
    inference_radius: int | float
        The inference radius for each local experts. We assume that all experts have the same inference radius.
    R: int | float, default 3
        A weight controlling the standard deviation of the Gaussian weights. The standard deviation will be given by
        the formula ``std = inference_radius / R``. The default value of 3 will place 99% of the Gaussian mass
        within the inference radius.

    Returns
    -------
    pandas dataframe
        A dataframe of glued predictions, whose columns contain (1) the prediction locations and (2) the glued variables.

    """
                              
    if isinstance(vars_to_glue, str):
        vars_to_glue = [vars_to_glue]

    preds = preds_df.copy(deep=True)

    # Compute Gaussian weights
    preds['total_weights'] = 1
    for (pred_col, xprt_col) in zip(pred_loc_cols, xprt_loc_cols):
        preds[f'weights_{xprt_col}'] = norm.pdf(preds[pred_col], preds[xprt_col], inference_radius/R)
        preds['total_weights'] *= preds[f'weights_{xprt_col}']

    # Multiply variables we wish to glue by Gaussian weights
    for var in vars_to_glue:
        preds[var] = preds[var] * preds['total_weights']

    # Compute weighted sum of variables, in addition to the total weights at each location
    glued_preds = preds[pred_loc_cols + ['total_weights'] + vars_to_glue].groupby(pred_loc_cols).sum()
    glued_preds = glued_preds.reset_index()
    
    # Normalise weighted sums with total weights
    for var in vars_to_glue:
        glued_preds[var] = glued_preds[var] / glued_preds['total_weights']
        
    return glued_preds.drop("total_weights", axis=1)


def get_smooth_params_config():

    config = get_config_from_sysargv()

    if config is None:

        config_file = get_parent_path("configs", "example_postprocessing.json")
        warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")
        with open(config_file, "r") as f:
            config = nested_dict_literal_eval(json.load(f))

        # HARDCODED: completely overriding reference result file
        config['result_file'] = get_parent_path("results", "example", "ABC_binned_example.h5")
        config['output_file'] = get_parent_path("results", "example", "ABC_binned_example.h5")

        cprint("example config being used:", c="BOLD")
        cprint(json.dumps(json_serializable(config), indent=4), c="HEADER")

    return config


if __name__ == "__main__":

    cprint('-' * 60, c="BOLD")
    cprint('-' * 60, c="BOLD")
    cprint("running smooth_hyperparameters, expect configuration (JSON) file to be provide as argument", c="OKBLUE")

    config = get_smooth_params_config()

    smooth_hyperparameters(**config)
