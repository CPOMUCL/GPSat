import types
import warnings
import xarray as xr
import numpy as np

import scipy.stats as scst

from GPSat.utils import config_func
from GPSat.decorators import timer
from GPSat.dataloader import DataLoader
class DataPrep:

    def __init__(self):
        """
        Constructor for the DataPrep class.

        Has class/static methods for preparing (e.g. binning) data
        """
        pass

    @classmethod
    @timer
    def bin_data_by(cls,
                    df,
                    col_funcs=None,
                    row_select=None,
                    by_cols=None,
                    val_col=None,
                    x_col='x', y_col='y',
                    x_range=None, y_range=None,
                    grid_res=None, bin_statistic="mean",
                    bin_2d=True,
                    limit=10000,
                    return_df=False,
                    verbose=False):
        """
        Class method to bin data by given columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing the data to be binned.

        col_funcs : dict, optional
            Dictionary with functions to be applied on the dataframe columns.

        row_select : dict, optional
            Dictionary with conditions to select rows of the dataframe.

        by_cols : str, list, tuple, optional
            Columns to be used for binning.

        val_col : str, optional
            Column with values to be used for binning.

        x_col : str, optional
            Name of the column to be used for x-axis, by default 'x'.

        y_col : str, optional
            Name of the column to be used for y-axis, by default 'y'.

        x_range : list, tuple, optional
            Range for the x-axis binning.

        y_range : list, tuple, optional
            Range for the y-axis binning.

        grid_res : float, optional
            Grid resolution for the binning process.

        bin_statistic : str or list, optional
            Statistic(s) to compute (default is 'mean').

        bin_2d : bool, default True
            if True bin data on a 2d grid, otherwise will perform 1d binning using only 'x'

        limit : int, optional
            Maximum number of unique values for the by_cols, by default 10000.

        return_df : bool, default False
            if True return results in a DataFrame, otherwise a Dataset (xarray)

        verbose : bool or int, optional
            If True or integer larger than 0, print information about process.

        Returns
        -------
        xarray.Dataset
            An xarray.Dataset containing the binned data.
        """

        # TODO: allow by_col to be missing - if it is could add a dummy column to df and then drop when not needed
        # TODO: this method may be more suitable in a different class - a DataPrep class
        # TODO: add print statements (given a verbose level)
        # TODO: grid_res should be in same dimensions as x,y

        # --
        # add / apply column functions
        # --

        if col_funcs is None:
            col_funcs = {}

        assert isinstance(col_funcs, dict), f"col_funcs must be a dictionary, got type: {type(col_funcs)}"

        # add columns
        # TODO: replace this with add_cols()
        for new_col, col_fun in col_funcs.items():

            # add new column
            if verbose >= 3:
                print(f"adding new_col: {new_col}")
            df[new_col] = config_func(df=df, **col_fun)

        # --
        # col checks
        # --

        if bin_2d is False:
            # if only binning in 1d, y_col isn't used - so set to x_col to pass checks below
            y_col = x_col

        assert by_cols is not None, f"by_col needs to be provided"
        if isinstance(by_cols, str):
            by_cols = [by_cols]
        assert isinstance(by_cols, (list, tuple)), f"by_cols must be list or tuple, got type: {type(by_cols)}"

        for bc in by_cols:
            assert bc in df, f"by_cols value: {bc} is not in df.columns: {df.columns}"

        assert val_col in df, f"val_col: {val_col} is not in df.columns: {df.columns}"
        assert x_col in df, f"x_col: {x_col} is not in df.columns: {df.columns}"
        assert y_col in df, f"y_col: {y_col} is not in df.columns: {df.columns}"

        # ----
        # select subset of rows
        # ----

        if row_select is not None:
            df = DataLoader.data_select(df, where=row_select)

        # get the common pairs
        bc_pair = df.loc[:, by_cols].drop_duplicates()

        assert len(bc_pair) < limit, f"number unique values of by_cols found in data: {len(bc_pair)} > limit: {limit} " \
                                     f"are you sure you want this many? if so increase limit"

        # allow for multiple bin statistics
        bin_statistic = bin_statistic if isinstance(bin_statistic, list) else [bin_statistic]

        da_list = []
        for idx, bcp in bc_pair.iterrows():

            # select data
            select = np.ones(len(df), dtype=bool)
            for bc in by_cols:
                select &= (df[bc] == bcp[bc]).values
            df_bin = df.loc[select, :]

            # store the 'by' coords
            by_coords = {bc: [bcp[bc]] for bc in by_cols}

            for bs_ix, bin_stat in enumerate(bin_statistic):

                b, crds = cls.bin_data(df_bin,
                                       x_range=x_range,
                                       y_range=y_range,
                                       grid_res=grid_res,
                                       x_col=x_col,
                                       y_col=y_col,
                                       val_col=val_col,
                                       bin_statistic=bin_stat,
                                       bin_2d=bin_2d,
                                       return_bin_center=True)

                # add extra dimensions to binned data
                b = b.reshape(b.shape + (1,) * len(by_cols))
                # store data in DataArray
                if bin_2d:
                    xc, yc = crds
                    # TODO: review y,x order - here assumes y is first dim. for symmetrical grids it doesn't matter
                    coords = {**{y_col: yc, x_col: xc}, **by_coords}
                else:
                    xc = crds
                    coords = {**{x_col: xc}, **by_coords}

                # --
                # determine name of DataArray
                # --
                # if there is only one bin_statistic then just use val_col - for backward compatibility
                if len(bin_statistic) == 1:
                    dataname = val_col
                else:
                    # if the bin_stat is a str e.g. 'mean', 'std', etc, append to val_col
                    if isinstance(bin_stat, str):
                        dataname = f"{val_col}_{bin_stat}"
                    # otherwise just use bin_stat name or index
                    else:
                        try:
                            # if bin_stat is a function try to get its name?
                            if isinstance(bin_stat, (types.FunctionType, types.BuiltinFunctionType)):
                                dataname = f"{val_col}_{bin_stat.__name__}"
                            else:
                                dataname = f"{val_col}_{bs_ix}"

                        except Exception as e:
                            print("in getting dataname received the following error:")
                            print(repr(e))
                            print("using index instead")
                            dataname = f"{val_col}_{bs_ix}"

                dims = [y_col, x_col] if bin_2d else [x_col]

                da = xr.DataArray(data=b,
                                  dims=dims + by_cols,
                                  coords=coords,
                                  name=dataname)
                da_list += [da]

        # combine into a single Dataset
        out = xr.combine_by_coords(da_list)

        return out.to_dataframe() if return_df else out

    @staticmethod
    def bin_data(
                 df,
                 x_range=None,
                 y_range=None,
                 grid_res=None,
                 x_col="x",
                 y_col="y",
                 val_col=None,
                 bin_statistic="mean",
                 bin_2d=True,
                 return_bin_center=True):
        """
        Bins the data contained within a DataFrame into a grid, optionally computes a statistic
        on the binned data, and returns the binned data along with the bin edges or centers.

        This method supports both 2D and 1D binning, allowing for various statistical computations
        on the binned values such as mean, median, count, etc.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data to be binned.
        x_range : tuple of float, optional
            The minimum and maximum values of the x-axis to be binned. If not provided, a default
            range is used.
        y_range : tuple of float, optional
            The minimum and maximum values of the y-axis to be binned. Only required for 2D binning.
            If not provided, a default range is used.
        grid_res : float
            The resolution of the grid in the same units as the x and y data. Defines the size of
            each bin.
        x_col : str, default 'x'
            The name of the column in `df` that contains the x-axis values.
        y_col : str, default 'y'
            The name of the column in `df` that contains the y-axis values. Ignored if `bin_2d` is False.
        val_col : str
            The name of the column in `df` that contains the values to be binned and aggregated.
        bin_statistic : str, default 'mean'
            The statistic to compute on the binned data. Can be 'mean', 'median', 'count', or any
            other statistic supported by `scipy.stats.binned_statistic` or
            `scipy.stats.binned_statistic_2d`.
        bin_2d : bool, default True
            If True, performs 2D binning using both x and y values. If False, performs 1D binning
            using only x values.
        return_bin_center : bool, default True
            If True, returns the center of each bin. If False, returns the edges of the bins.

        Returns
        -------
        binned_data : numpy.ndarray
            An array of the binned and aggregated data. The shape of the array depends on the
            binning dimensions and the grid resolution.
        x_bin : numpy.ndarray
            An array of the x-axis bin centers or edges, depending on the value of
            `return_bin_center`.
        y_bin : numpy.ndarray, optional
            An array of the y-axis bin centers or edges, only returned if `bin_2d` is True and
            `return_bin_center` is specified.

        Raises
        ------
        AssertionError
            If `val_col` or `grid_res` is not specified, or if the DataFrame `df` is empty.
            Also raises an error if the provided `x_range` or `y_range` are invalid or if
            the specified column names are not present in `df`.

        Notes
        -----
        - The default `x_range` and `y_range` are set to [-4500000.0, 4500000.0] if not provided.
        - This method requires that `val_col` and `grid_res` be explicitly provided.
        - The binning process is influenced by the `bin_statistic` parameter, which determines
          how the values in each bin are aggregated.
        - When `bin_2d` is False, `y_col` is ignored and only `x_col` and `val_col` are used for binning.
        - The method ensures that the `x_col`, `y_col`, and `val_col` exist in the DataFrame `df`.
        """
        # TODO: allow for the x,y locations return to be, instead of the bin center,
        #  be the average location of values in binned
        # TODO: review doc string
        # TODO: move defaults out of bin_data to bin_data_by?
        # TODO: double check get desired shape, dim alignment if x_range != y_range

        # ---
        # check inputs, handle defaults

        assert val_col is not None, "val_col - the column containing values to bin cannot be None"
        assert grid_res is not None, "grid_res is None, must be supplied - expressed in km"
        assert len(df) > 0, f"dataframe (df) provide must have len > 0"

        # if not binning by 2d default to 2d
        # - data from the 'y' dimension is not needed, set to 'x' and then ignore
        if not bin_2d:
            y_col = x_col

        if x_range is None:
            x_range = [-4500000.0, 4500000.0]
            print(f"x_range, not provided, using default: {x_range}")
        assert x_range[0] < x_range[1], f"x_range should be (min, max), got: {x_range}"

        if y_range is None:
            y_range = [-4500000.0, 4500000.0]
            if bin_2d:
                print(f"y_range, not provided, using default: {y_range}")
        assert y_range[0] < y_range[1], f"y_range should be (min, max), got: {y_range}"

        assert len(x_range) == 2, f"x_range expected to be len = 2, got: {len(x_range)}"
        assert len(y_range) == 2, f"y_range expected to be len = 2, got: {len(y_range)}"

        # if grid_res is None:
        #     grid_res = 50
        #     print(f"grid_res, not provided, using default: {grid_res}")

        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

        # number of bin (edges)
        n_x = ((x_max - x_min) / grid_res) + 1
        n_y = ((y_max - y_min) / grid_res) + 1
        n_x, n_y = int(n_x), int(n_y)

        # bin parameters
        assert x_col in df, f"x_col: {x_col} is not in df columns: {df.columns}"
        assert y_col in df, f"y_col: {y_col} is not in df columns: {df.columns}"
        assert val_col in df, f"val_col: {val_col} is not in df columns: {df.columns}"

        # NOTE: x will be dim 1, y will be dim 0
        x_edge = np.linspace(x_min, x_max, int(n_x))
        y_edge = np.linspace(y_min, y_max, int(n_y))

        # extract values
        x_in, y_in, vals = df[x_col].values, df[y_col].values, df[val_col].values

        # apply binning
        if bin_2d:
            binned = scst.binned_statistic_2d(x_in, y_in, vals,
                                              statistic=bin_statistic,
                                              bins=[x_edge,
                                                    y_edge],
                                              range=[[x_min, x_max], [y_min, y_max]])

            # get the average value of the x and y coordinates in the bin
            # x_mean = scst.binned_statistic_2d(x_in, y_in, x_in,
            #                                   statistic=bin_statistic,
            #                                   bins=[x_edge,
            #                                         y_edge],
            #                                   range=[[x_min, x_max], [y_min, y_max]])
            # y_mean = scst.binned_statistic_2d(x_in, y_in, y_in,
            #                                   statistic=bin_statistic,
            #                                   bins=[x_edge,
            #                                         y_edge],
            #                                   range=[[x_min, x_max], [y_min, y_max]])

        else:
            binned = scst.binned_statistic(x_in, vals,
                                           statistic=bin_statistic,
                                           bins=x_edge,
                                           range=[x_min, x_max])

            # get the average value of the x-coordinate in the bin
            # x_mean = scst.binned_statistic(x_in, x_in,
            #                                statistic='mean',
            #                                bins=x_edge,
            #                                range=[x_min, x_max])

        # why not use the edge values from binned?
        # TODO: validate binned edges (in binned objects) are the same as below - write unit test to validate
        xy_out = x_edge, y_edge
        # return the bin centers, instead of the edges?
        if return_bin_center:
            x_cntr, y_cntr = x_edge[:-1] + np.diff(x_edge) / 2, y_edge[:-1] + np.diff(y_edge) / 2
            xy_out = x_cntr, y_cntr

        # TODO: if output is transpose, should the x,y (edges or centers) be swapped?
        if bin_2d:
            return binned[0].T, (xy_out[0], xy_out[1])
        else:
            return binned[0].T, xy_out[0]



if __name__ == "__main__":

    pass
