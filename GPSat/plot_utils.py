import copy
import re
import warnings

import numpy as np
import seaborn as sns
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt

from GPSat.decorators import timer
from GPSat.dataloader import DataLoader
from GPSat.utils import pretty_print_class, dataframe_to_2d_array, EASE2toWGS84_New, \
    get_weighted_values, cprint

# 'optional' / conda specific packages
try:
    # NOTE: as cartopy is not a required package
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
except ImportError as e:
    print("ImportError for cartopy package, if want to plot data on maps install with: conda install cartopy\n"
          "requires activated conda environment")
    print(e)
    ccrs = None
    cfeat = None

try:
    from global_land_mask import globe as globe_mask
except ModuleNotFoundError:
    cprint("could not import global_land_mask package, won't reduce grid points to just those over ocean.\ninstall with: pip install global-land-mask", c="HEADER")
    globe_mask = None


@timer
def plot_pcolormesh(ax, lon, lat, plot_data,
                    fig=None,
                    title=None,
                    vmin=None,
                    vmax=None,
                    qvmin=None,
                    qvmax=None,
                    cmap='YlGnBu_r',
                    cbar_label=None,
                    scatter=False,
                    extent=None,
                    ocean_only=False,
                    **scatter_args):
    # TODO: finish with scatter option
    # TODO: deal with ShapelyDeprecationWarning (geoms?)
    # TODO: add qvmin, qvmax - quantile based vmin/vmax

    # ax = axs[j]
    ax.coastlines(resolution='50m', color='white')
    ax.add_feature(cfeat.LAKES, color='white', alpha=.5)
    ax.add_feature(cfeat.LAND, color=(0.8, 0.8, 0.8))
    extent = [-180, 180, 60, 90] if extent is None else extent
    ax.set_extent(extent, ccrs.PlateCarree())  # lon_min,lon_max,lat_min,lat_max

    if title:
        ax.set_title(title)

    if ocean_only:
        if globe_mask is None:
            warnings.warn(f"ocean_only={ocean_only}, however globe_mask is missing, "
                          f"install with pip install global-land-mask")
        else:
            is_in_ocean = globe_mask.is_ocean(lat, lon)
            # copy, just to be safe
            plot_data = copy.copy(plot_data)
            plot_data[~is_in_ocean] = np.nan

    if qvmin is not None:
        if vmin is not None:
            warnings.warn("both qvmin and vmin are supplied, only using qvmin")
        assert (qvmin >= 0) & (qvmin <= 1.0), f"qvmin: {qvmin}, needs to be in [0,1]"
        vmin = np.nanquantile(plot_data, q=qvmin)

    if qvmax is not None:
        if vmax is not None:
            warnings.warn("both qvmax and vmax are supplied, only using qvmax")
        assert (qvmax >= 0) & (qvmax <= 1.0), f"qvmax: {qvmax}, needs to be in [0,1]"
        vmax = np.nanquantile(plot_data, q=qvmax)

    if (vmin is not None) & (vmax is not None):
        assert vmin <= vmax, f"vmin: {vmin} > vmax: {vmax}"

    if not scatter:
        s = ax.pcolormesh(lon, lat, plot_data,
                          cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(),
                          linewidth=0,
                          shading="auto",# to remove DeprecationWarning
                          rasterized=True)
    else:
        non_nan = ~np.isnan(plot_data)
        s = ax.scatter(lon[non_nan],
                       lat[non_nan],
                       c=plot_data[non_nan],
                       cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(),
                       linewidth=0,
                       rasterized=True,
                       **scatter_args)

    if fig is not None:
        cbar = fig.colorbar(s, ax=ax, orientation='horizontal', pad=0.03, fraction=0.03)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=14)
        cbar.ax.tick_params(labelsize=14)

@timer
def plot_hist(ax, data,
              title="Histogram / Density",
              ylabel=None,
              xlabel=None,
              select_bool=None,
              stats_values=None,
              stats_loc=(0.2, 0.9),
              drop_nan_inf=True,
              q_vminmax=None):

    hist_data = data if select_bool is None else data[select_bool]

    # drop any nan or inf?
    if drop_nan_inf:
        hist_data = hist_data[~np.isnan(hist_data)]
        hist_data = hist_data[~np.isinf(hist_data)]


    # trim data that is plotted (won't affect stats)
    if q_vminmax is not None:
        assert isinstance(q_vminmax, (list, tuple, np.ndarray)), \
            f"q_vminmax expected to be list, tuple or array, got: {type(q_vminmax)}"
        assert len(q_vminmax) == 2, f"len(q_vminmax) expected to be 2, got: {len(q_vminmax)}"
        vmin, vmax = np.nanquantile(hist_data, q=list(q_vminmax))
        hist_data = hist_data[(hist_data >= vmin) & (hist_data <= vmax)]

    sns.histplot(data=hist_data, kde=True, ax=ax, rasterized=True)
    ax.set(ylabel=ylabel)
    ax.set(xlabel=xlabel)
    ax.set(title=title)

    # provide stats if stats values is not None
    if stats_values is not None:
        # drop any nan or inf?
        if drop_nan_inf:
            data = data[~np.isnan(data)]
            data = data[~np.isinf(data)]

        stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "skew": skew(data),
            "kurtosis": kurtosis(data),
            "num obs": len(data),
            "max": np.max(data),
            "min": np.min(data)
        }

        stats_values = [stats_values] if isinstance(stats_values, str) else stats_values
        for sv in stats_values:
            assert sv in stats, f"stats_values: {sv} not in stats: {list(stats.keys)}"
        stats = {_: stats[_] for _ in stats_values}
        stats_str = "\n".join([f"{kk}: {vv:.2f}" if isinstance(vv, float) else f"{kk}: {vv:d}"
                               for kk, vv in stats.items()])
        ax.text(stats_loc[0], stats_loc[1], stats_str,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)


def get_projection(projection=None):

    # projection
    if projection is None:
        projection = ccrs.NorthPolarStereo()
    elif isinstance(projection, ccrs.Projection):
        pass
    elif isinstance(projection, str):
        if re.search("north", projection, re.IGNORECASE):
            projection = ccrs.NorthPolarStereo()
        elif re.search("south", projection, re.IGNORECASE):
            projection = ccrs.SouthPolarStereo()
        else:
            raise NotImplementedError(f"projection provide as str: {projection}, not implemented")

    return projection

@timer
def plot_xy(ax, x, y,
            title=None,
            y_label=None,
            x_label=None,
            xtick_rotation=45,
            scatter=False,
            **kwargs):

    if scatter:
        ax.scatter(x, y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)

    if title:
        ax.set_title(title)

    if y_label:
        ax.set_ylabel(y_label)

    if x_label:
        ax.set_xlabel(x_label)

    ax.tick_params(axis='x', rotation=xtick_rotation)


def plot_xy_from_results_data(ax, dfs, table, x_col, y_col,
                              load_kwargs=None, plot_kwargs=None, verbose=False, **kwargs):
    # NOTE: fig not used
    # TODO: be more explicit with input parma

    if load_kwargs is None:
        load_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # kwargs aren't used, just being lazy to avoid popping out unused kwargs from configs
    if (len(kwargs) > 0) & (verbose):
        print("unused kwargs")
        print(kwargs)

    # load data
    plt_data = DataLoader.load(dfs[table],
                               **load_kwargs)
    # get inputs
    x, y = plt_data[x_col], plt_data[y_col]

    # plot on given subplot
    plot_xy(ax=ax, x=x, y=y, **plot_kwargs)


def plot_hist_from_results_data(ax, dfs, table, val_col,
                                load_kwargs=None, plot_kwargs=None, verbose=False, **kwargs):
    # NOTE: fig not used
    # TODO: be more explicit with input parma

    if load_kwargs is None:
        load_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # kwargs aren't used, just being lazy to avoid popping out unused kwargs from configs
    if (len(kwargs) > 0) & (verbose):
        print("unused kwargs")
        print(kwargs)

    # load data
    plt_data = DataLoader.load(dfs[table],
                               **load_kwargs)
    #
    plot_hist(ax=ax,
              data=plt_data[val_col],  # plt_df[val_col].values,
              **plot_kwargs)


def plot_pcolormesh_from_results_data(ax, dfs, table, val_col,
                                      lon_col=None, lat_col=None,
                                      x_col=None, y_col=None, lat_0=90, lon_0=0,
                                      fig=None,
                                      load_kwargs=None,
                                      plot_kwargs=None,
                                      weighted_values_kwargs=None,
                                      verbose=False, **kwargs):
    # x_col, y_col used to extract 2d array, expected to be regularly spaced

    if load_kwargs is None:
        load_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # kwargs aren't used, just being lazy to avoid popping out unused kwargs from configs
    if (len(kwargs) > 0) & (verbose):
        print("unused kwargs")
        print(kwargs)

    # load data
    plt_data = DataLoader.load(dfs[table],
                               **load_kwargs)
    # check columns are in data
    for _ in [x_col, y_col, lon_col, lat_col, val_col]:
        if _ is not None:
            assert _ in plt_data, f"'{_}' (column) not in plot_data"

    # get a weighted combination of values?
    if weighted_values_kwargs is not None:
        assert isinstance(weighted_values_kwargs, dict)
        plt_data = get_weighted_values(df=plt_data, **weighted_values_kwargs)

        # HACK: get_weighted_values drops columns, apply column functions if need be
        plt_data = DataLoader._modify_df(plt_data,
                                         col_funcs=load_kwargs.get("col_funcs", None),
                                         verbose=verbose)


    # 2d array
    if not plot_kwargs.get("scatter", False):
        assert (x_col is not None) & (y_col is not None), f"plotting 2d array requires " \
                                                          f"x_col: {x_col} and y_col: {y_col} to both not be None"
        val2d, x_grid, y_grid = dataframe_to_2d_array(df=plt_data,
                                                      x_col=x_col,
                                                      y_col=y_col,
                                                      val_col=val_col)
        # convert the x,y coords to lon lat coords
        lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid, lat_0=lat_0, lon_0=lon_0)

        plot_pcolormesh(ax=ax,
                        lon=lon_grid,
                        lat=lat_grid,
                        plot_data=val2d,
                        fig=fig,
                        **plot_kwargs)

    # scatter plot
    else:
        assert (lon_col is not None) & (lat_col is not None), f"scatter plot requires " \
                                                              f"lon_col: {lon_col} and " \
                                                              f"lat_col: {lat_col} to both not be None"

        plot_pcolormesh(ax=ax,
                        lon=plt_data[lon_col].values,
                        lat=plt_data[lat_col].values,
                        plot_data=plt_data[val_col].values,
                        fig=fig,
                        **plot_kwargs)


def plot_gpflow_minimal_example(model: object, model_init: object = None, opt_params: object = None, pred_params: object = None) -> object:
    """
    Run a basic usage example for a given model.
    Model will be initialised, parameters will be optimised and predictions will be made
    for the minimal model example found (as of 2023-05-04):
    
    https://gpflow.github.io/GPflow/2.8.0/notebooks/getting_started/basic_usage.html

    Methods called are: optimise_parameters, predict, get_parameters

    Predict expected to return a dict with 'f*', 'f*_var' and 'y_var' as np.arrays

    Parameters
    ----------
    model: any model inherited from BaseGPRModel
    model_init: dict or None, default None
        dict of parameters to be provided when model is initialised. If None default parameters are used
    opt_params: dict or None, default None
        dict of parameters to be passed to optimise_parameter method. If None default parameters are used
    pred_params: dict or None, default None
        dict of parameters to be passed to predict method. If None default parameters are used


    Returns
    -------
    tuple:
        predictions dict
        parameters dict

    """
    
    # --
    # check additional params
    # --

    model_init = {} if model_init is None else model_init
    assert isinstance(model_init, dict), f"model_init expected to be dict, got: {type(model_init)}"

    opt_params = {} if opt_params is None else opt_params
    assert isinstance(opt_params, dict), f"opt_params expected to be dict, got: {type(model_init)}"

    pred_params = {} if pred_params is None else pred_params
    assert isinstance(pred_params, dict), f"pred_params expected to be dict, got: {type(model_init)}"

    # --
    # toy data
    # ---

    X = np.array(
        [
            [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
            [0.171], [0.889], [0.243], [0.028],
        ]
    )
    Y = np.array(
        [
            [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
            [1.30], [4.00], [3.82],
        ]
    )

    # initialise the model

    m = model(coords=X, obs=Y, **model_init)

    # test points must be of shape (N, D)
    Xplot = np.linspace(-0.1, 1.1, 100)[:, None]

    # optimise
    optimised = m.optimise_parameters(**opt_params)

    print(f"optimised: {optimised}")

    params = m.get_parameters()
    print(f"parameters:\n{params}")

    # predict - without optimising
    preds = m.predict(Xplot)

    # extract predictions
    f_mean, f_var, y_var = preds['f*'], preds['f*_var'], preds['y_var']

    # predict mean and variance of latent GP at test points
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    y_lower = f_mean - 1.96 * np.sqrt(y_var)
    y_upper = f_mean + 1.96 * np.sqrt(y_var)

    # --
    # plot values
    # --

    plt.plot(X, Y, "kx", mew=2, label="input data")
    plt.plot(Xplot, f_mean, "-", color="C0", label="(predicted) mean")
    plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
    plt.plot(Xplot, f_upper, "--", color="C0")
    plt.fill_between(
        Xplot[:, 0], f_lower, f_upper, color="C0", alpha=0.1
    )
    plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
    plt.plot(Xplot, y_upper, ".", color="C0")
    plt.fill_between(
        Xplot[:, 0], y_lower, y_upper, color="C0", alpha=0.1
    )
    plt.legend()
    plt.title(pretty_print_class(m))
    plt.show()

    return preds, params


def plots_from_config(plot_configs, dfs, num_plots_row_col_size, suptitle=""):
    plt_idx = 1

    rcs = num_plots_row_col_size[len(plot_configs)]
    nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
    fig = plt.figure(figsize=fig_size)

    fig.suptitle(suptitle, y=1.0)

    for p in plot_configs:

        # create a new subplot
        subplot_kwargs = p.get("subplot_kwargs", {})
        if "projection" in subplot_kwargs:
            subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])

        ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)

        assert 'plot_type' in p, "'plot_type' is not specified in plot_config"
        if p['plot_type'] == "plot_xy":
            plot_xy_from_results_data(ax=ax, dfs=dfs, **p)
        elif p['plot_type'] == "hist":
            plot_hist_from_results_data(ax=ax, dfs=dfs, **p)
        elif p['plot_type'] == "heatmap":
            plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **p)
        else:
            raise NotImplementedError(f"plot_type: '{p['plot_type']}' is not implemented")
        plt_idx += 1

    plt.tight_layout()
    # plt.show()

    return fig


if __name__ == "__main__":

    pass
