# %% [markdown]
#### Inline Example of Local Expert 'Optimal Interpolation' on Satellite Data


## Using Colab? Then clone and install
# %%
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import subprocess
    import os
    import re

    # change to working directory
    work_dir = "/content"

    assert os.path.exists(work_dir), f"workspace directory: {work_dir} does not exist"
    os.chdir(work_dir)

    # clone repository
    command = "git clone https://github.com/CPOMUCL/GPSat.git"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    print(result.stdout)

    repo_dir = os.path.join(work_dir, "GPSat")

    print(f"changing directory to: {repo_dir}")
    os.chdir(repo_dir)

    # exclude certain requirements if running on colab - namely avoid installing/upgrading tensorflow
    new_req = []
    with open(os.path.join(repo_dir, "requirements.txt"), "r") as f:
        for line in f.readlines():
            # NOTE: here also removing numpy requirement
            if re.search("^tensorflow|^numpy", line):
                new_req.append("#" + line)
            else:
                new_req.append(line)

    # create a colab specific requirements file
    with open(os.path.join(repo_dir, "requirements_colab.txt"), "w") as f:
        f.writelines(new_req)

    # install the requirements
    command = "pip install -r requirements_colab.txt"
    with subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        # Stream the standard output in real-time
        for line in proc.stdout:
            print(line, end='')

    # install the GPSat pacakge in editable mode
    command = "pip install -e ."
    with subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        # Stream the standard output in real-time
        for line in proc.stdout:
            print(line, end='')


# %% [markdown]
##  Import Packages
# %%

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from global_land_mask import globe
from GPSat import get_data_path, get_parent_path
from GPSat.dataprepper import DataPrep
from GPSat.dataloader import DataLoader
from GPSat.utils import stats_on_vals, WGS84toEASE2, EASE2toWGS84, cprint, grid_2d_flatten, get_weighted_values
from GPSat.local_experts import LocalExpertOI, get_results_from_h5file
from GPSat.plot_utils import plot_wrapper, plot_pcolormesh, get_projection, plot_pcolormesh_from_results_data, plot_hyper_parameters
from GPSat.postprocessing import smooth_hyperparameters


# %% [markdown]
## Parameters
# %%

# NOTE: there are parameters values that are set inline in the cells below

# lat,lon center (origin) used for converting between WGS84 and EASE2 projections
lat_0 = 90
lon_0 = 0

# expert location parameters
# spacing between experts (laid out on a grid), in meters
expert_spacing = 200_000
# range of experts, from origin, in meters
# expert_x_range = [-750_000.0, 1000_000.0]
# expert_y_range = [-500_000.0, 1250_000.0]
expert_x_range = [-500_000, 500_000]
expert_y_range = [-500_000, 500_000]

# prediction spacing
# (below predictions same range as experts)
pred_spacing = 5_000


# model parameters
# Set training and inference radius
# - distance observations need to be away from expert locations to be included in training
training_radius = 300_000  # 300km
# - distance prediction locations need to be away from expert locations in order of predictions to be made
inference_radius = 200_000  # 200km


# plotting
# extent = [lon min, lat max, lat min, lat max]
extent = [-180, 180, 60, 90]

# which projection to use: "north" or "south"
projection = "north"

# %% [markdown]
##  read in raw data

# add each key in col_func as a column, using a specified function + arguments
# values are unpacked and passed to GPSat.utils.config_func
# %%


df = DataLoader.read_flat_files(file_dirs=get_data_path("example"),
                                file_regex="_RAW\.csv$",
                                col_funcs={
                                    "source": {
                                        "func": lambda x: re.sub('_RAW.*$', '', os.path.basename(x)),
                                        "filename_as_arg": True
                                    }
                                })

# convert lon, lat, datetime to x, y, t - to be used as the coordinate space
df['x'], df['y'] = WGS84toEASE2(lon=df['lon'], lat=df['lat'], lat_0=lat_0, lon_0=lon_0)
df['t'] = df['datetime'].values.astype("datetime64[D]").astype(float)

# %% [markdown]
## stats on data
# %%

print("*" * 20)
print("summary / stats table on metric (use for trimming)")

val_col = 'z'
vals = df[val_col].values
stats_df = stats_on_vals(vals=vals, name=val_col,
                         qs=[0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99])

print(stats_df)

# %% [markdown]
## visualise data
# %%

# plot observations and histogram
fig, stats_df = plot_wrapper(plt_df=df,
                             val_col=val_col,
                             max_obs=500_000,
                             vmin_max=[-0.1, 0.5],
                             projection=projection,
                             extent=extent)

plt.show()


# %% [markdown]
## bin raw data
# bin by date, source - returns a DataSet
# %%

bin_ds = DataPrep.bin_data_by(df=df.loc[(df['z'] > -0.35) & (df['z'] < 0.65)],
                              by_cols=['t', 'source'],
                              val_col=val_col,
                              x_col='x',
                              y_col='y',
                              grid_res=50_000,
                              x_range=[-4500_000.0, 4500_000.0],
                              y_range=[-4500_000.0, 4500_000.0])

# convert bin data to DataFrame
# - removing all the nans that would be added at grid locations away from data
bin_df = bin_ds.to_dataframe().dropna().reset_index()

# %% [markdown]
## plot binned data
# %%

# this will plot all observations, some on top of each other
bin_df['lon'], bin_df['lat'] = EASE2toWGS84(bin_df['x'], bin_df['y'],
                                            lat_0=lat_0, lon_0=lon_0)

# plot observations and histogram
fig, stats_df = plot_wrapper(plt_df=bin_df,
                             val_col=val_col,
                             max_obs=500_000,
                             vmin_max=[-0.1, 0.5],
                             projection=projection,
                             extent=extent)

plt.show()


# %% [markdown]
## expert locations
# on evenly spaced grid
# %%

xy_grid = grid_2d_flatten(x_range=expert_x_range,
                          y_range=expert_y_range,
                          step_size=expert_spacing)

# store in dataframe
eloc = pd.DataFrame(xy_grid, columns=['x', 'y'])

# add a time coordinate
eloc['t'] = np.floor(df['t'].mean())

# %% [markdown]
## plot expert locations
# %%

eloc['lon'], eloc['lat'] = EASE2toWGS84(eloc['x'], eloc['y'],
                                        lat_0=lat_0, lon_0=lon_0)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection(projection))

plot_pcolormesh(ax=ax,
                lon=eloc['lon'],
                lat=eloc['lat'],
                plot_data=eloc['t'],
                title="expert locations",
                scatter=True,
                s=20,
                fig=fig,
                extent=extent)

plt.tight_layout()
plt.show()

# %% [markdown]
## prediction locations
# %%

pred_xy_grid = grid_2d_flatten(x_range=expert_x_range,
                               y_range=expert_y_range,
                               step_size=pred_spacing)

# store in dataframe
# NOTE: the missing 't' coordinate will be determine by the expert location
# - alternatively the prediction location can be specified
ploc = pd.DataFrame(pred_xy_grid, columns=['x', 'y'])

ploc['lon'], ploc['lat'] = EASE2toWGS84(ploc['x'], ploc['y'],
                                        lat_0=lat_0, lon_0=lon_0)

# identify if a position is in the ocean (water) or not
ploc["is_in_ocean"] = globe.is_ocean(ploc['lat'], ploc['lon'])

# keep only prediction locations in ocean
ploc = ploc.loc[ploc["is_in_ocean"]]

# %% [markdown]
## plot prediction locations
# %%

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection(projection))

plot_pcolormesh(ax=ax,
                lon=ploc['lon'],
                lat=ploc['lat'],
                plot_data=np.full(len(ploc), 1.0),  # np.arange(len(ploc)),
                title="prediction locations",
                scatter=True,
                s=0.1,
                # fig=fig,
                extent=extent)

plt.tight_layout()
plt.show()

# %% [markdown]
## configurations:
# %%

# observation data
data = {
    "data_source": bin_df,
    "obs_col": "z",
    "coords_col": ["x", "y", "t"],
    # selection criteria used for each local expert
    "local_select": [
        {
            "col": "t",
            "comp": "<=",
            "val": 4
        },
        {
            "col": "t",
            "comp": ">=",
            "val": -4
        },
        {
            "col": [
                "x",
                "y"
            ],
            "comp": "<",
            "val": training_radius
        }
    ]
}

# local expert locations
local_expert = {
    "source": eloc
}

# model
model = {
    "oi_model": "GPflowGPRModel",
    "init_params": {
        # scale (divide) coordinates
        "coords_scale": [50000, 50000, 1],
        # can specify initial parameters values for model:
        # "noise_variance": 0.10,
        # "kernel_kwargs": {
        #     "lengthscales": [2.0, 2.0, 1.0],
        #     "variance": 0.05
        # }
    },
    # keyword arguments to be passed into each model/local expert's optimise_parameters method
    "optim_kwargs": {
        # parameters to be fixed (not trainable)
        # "fixed_params": ["likelihood_variance"]
    },
    "constraints": {
        # lengthscales - same order coord_col (see data)
        # - given in unscaled units
        "lengthscales": {
            "low": [1e-08, 1e-08, 1e-08],
            "high": [600000, 600000, 9]
        },
        "likelihood_variance": {
            "low": 0.00125,
            "high": 0.01
        }
    }
}

# prediction locations
pred_loc = {
    "method": "from_dataframe",
    "df": ploc,
    "max_dist": inference_radius
}

# %% [markdown]
## Local Expert OI

# if process falls over here when calling run(), try: Runtime -> "Restart and run all"
# %%

locexp = LocalExpertOI(expert_loc_config=local_expert,
                       data_config=data,
                       model_config=model,
                       pred_loc_config=pred_loc)

# run optimal interpolation
# - no predictions locations supplied
store_path = get_parent_path("results", "inline_example.h5")

# for the purposes of a simple example, if store_path exists: delete it
if os.path.exists(store_path):
    cprint(f"removing: {store_path}", "FAIL")
    os.remove(store_path)

# run optimal interpolation
locexp.run(store_path=store_path,
           optimise=True,
           check_config_compatible=False)

# %% [markdown]
# results are store in hdf5
# %%

# extract, store in dict
dfs, oi_config = get_results_from_h5file(store_path)

print(f"tables in results file: {list(dfs.keys())}")

# %% [markdown]
# Plot Hyper Parameters
# %%

# a template to be used for each created plot config
plot_template = {
    "plot_type": "heatmap",
    "x_col": "x",
    "y_col": "y",
    # use a northern hemisphere projection, centered at (lat,lon) = (90,0)
    "subplot_kwargs": {"projection": projection},
    "lat_0": lat_0,
    "lon_0": lon_0,
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": True,
    },
    # lat/lon_col needed if scatter = True
    # TODO: remove the need for this
    "lat_col": "lat",
    "lon_col": "lon",
}

fig = plot_hyper_parameters(dfs,
                            coords_col=oi_config[0]['data']['coords_col'],  # ['x', 'y', 't']
                            row_select=None,  # this could be used to select a specific date in results data
                            table_names=["lengthscales", "kernel_variance", "likelihood_variance"],
                            plot_template=plot_template,
                            plots_per_row=3,
                            suptitle="hyper params",
                            qvmin=0.01,
                            qvmax=0.99)

plt.show()

# %% [markdown]
# Smooth Hyper Parameters
# %%

smooth_config = {
    # get hyper parameters from the previously stored results
    "result_file": store_path,
    # store the smoothed hyper parameters in the same file
    "output_file": store_path,
    # get the hyper params from tables ending with this suffix ("" is default):
    "reference_table_suffix": "",
    # newly smoothed hyper parameters will be store in tables ending with table_suffix
    "table_suffix": "_SMOOTHED",
    # dimension names to smooth over
    "xy_dims": [
        "x",
        "y"
    ],
    # parameters to smooth
    "params_to_smooth": [
        "lengthscales",
        "kernel_variance",
        "likelihood_variance"
    ],
    # length scales for the kernel smoother in each dimension
    # - as well as any min/max values to apply
    "smooth_config_dict": {
        "lengthscales": {
            "l_x": 200_000,
            "l_y": 200_000
        },
        "likelihood_variance": {
            "l_x": 200_000,
            "l_y": 200_000,
            "max": 0.3
        },
        "kernel_variance": {
            "l_x": 200_000,
            "l_y": 200_000,
            "max": 0.1
        }
    },
    "save_config_file": True
}

smooth_result_config_file = smooth_hyperparameters(**smooth_config)

# modify the model configuration to include "load_params"
model_load_params = model.copy()
model_load_params["load_params"] = {
    "file": store_path,
    "table_suffix": smooth_config["table_suffix"]
}

locexp_smooth = LocalExpertOI(expert_loc_config=local_expert,
                              data_config=data,
                              model_config=model_load_params,
                              pred_loc_config=pred_loc)

# run optimal interpolation (again)
# - this time don't optimise hyper parameters, but make predictions
# - store results in new tables ending with '_SMOOTHED'
locexp_smooth.run(store_path=store_path,
                  optimise=False,
                  predict=True,
                  table_suffix=smooth_config['table_suffix'],
                  check_config_compatible=False)

# %% [markdown]
## Plot Smoothed Hyper Parameters
# %%
# extract, store in dict
dfs, oi_config = get_results_from_h5file(store_path)

fig = plot_hyper_parameters(dfs,
                            coords_col=oi_config[0]['data']['coords_col'],  # ['x', 'y', 't']
                            row_select=None,
                            table_names=["lengthscales", "kernel_variance", "likelihood_variance"],
                            table_suffix=smooth_config["table_suffix"],
                            plot_template=plot_template,
                            plots_per_row=3,
                            suptitle="smoothed hyper params",
                            qvmin=0.01,
                            qvmax=0.99)

plt.tight_layout()
plt.show()

# %% [markdown]
## get weighted combinations predictions and plot
# %%

plt_data = dfs["preds" + smooth_config["table_suffix"]]

weighted_values_kwargs = {
    "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
    "dist_to_col": ["x", "y", "t"],
    "val_cols": ["f*", "f*_var"],
    "weight_function": "gaussian",
    "lengthscale": inference_radius/2
}
plt_data = get_weighted_values(df=plt_data, **weighted_values_kwargs)

plt_data['lon'], plt_data['lat'] = EASE2toWGS84(plt_data['pred_loc_x'], plt_data['pred_loc_y'])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection(projection))
plot_pcolormesh_from_results_data(ax=ax,
                                  dfs={"preds": plt_data},
                                  table='preds',
                                  val_col="f*",
                                  x_col='pred_loc_x',
                                  y_col='pred_loc_y',
                                  fig=fig)
plt.tight_layout()
plt.show()

