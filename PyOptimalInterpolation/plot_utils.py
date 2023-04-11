
import re
import numpy as np
import seaborn as sns
from scipy.stats import skew, kurtosis

from PyOptimalInterpolation.dataloader import DataLoader

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


def plot_pcolormesh(ax, lon, lat, plot_data,
                    fig=None,
                    title=None,
                    vmin=None,
                    vmax=None,
                    cmap='YlGnBu_r',
                    cbar_label=None,
                    scatter=False,
                    extent=None,
                    **scatter_args):
    # TODO: finish with scatter option
    # TODO: deal with ShapelyDeprecationWarning (geoms?)

    # ax = axs[j]
    ax.coastlines(resolution='50m', color='white')
    ax.add_feature(cfeat.LAKES, color='white', alpha=.5)
    ax.add_feature(cfeat.LAND, color=(0.8, 0.8, 0.8))
    extent = [-180, 180, 60, 90] if extent is None else extent
    ax.set_extent(extent, ccrs.PlateCarree())  # lon_min,lon_max,lat_min,lat_max

    if title:
        ax.set_title(title)

    if not scatter:
        s = ax.pcolormesh(lon, lat, plot_data,
                          cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(),
                          linewidth=0,
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


def plot_hist(ax, data,
              title="Histogram / Density",
              ylabel=None,
              xlabel=None,
              select_bool=None,
              stats_values=None,
              stats_loc=(0.2, 0.9),
              q_vminmax=None):

    hist_data = data if select_bool is None else data[select_bool]

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


def plot_pcolormesh_from_results_data(ax, dfs, table, lon_col, lat_col, val_col,
                                      fig=None, load_kwargs=None, plot_kwargs=None, verbose=False, **kwargs):
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
    for _ in [lon_col, lat_col, val_col]:
        assert _ in plt_data, f"'{_}' (column) not in plot_data"

    plot_pcolormesh(ax=ax,
                    lon=plt_data[lon_col].values,
                    lat=plt_data[lat_col].values,
                    plot_data=plt_data[val_col].values,
                    fig=fig,
                    **plot_kwargs)




if __name__ == "__main__":

    pass
