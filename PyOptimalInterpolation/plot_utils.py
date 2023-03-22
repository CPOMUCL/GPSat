
import numpy as np
import seaborn as sns
from scipy.stats import skew, kurtosis

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


if __name__ == "__main__":

    pass
