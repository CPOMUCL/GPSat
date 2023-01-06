
import warnings
import xarray as xr
import numpy as np

import scipy.stats as scst

from PyOptimalInterpolation.utils import config_func
from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.dataloader import DataLoader
class DataPrep:

    def __init__(self):
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
                    limit=10000,
                    verbose=False):

        # TODO: this method may be more suitable in a different class - a DataPrep class
        # TODO: add doc string
        # TODO: add print statements (given a verbose level)
        # TODO: grid_res should be in same dimensions as x,y

        # --
        # add / apply column functions
        # --

        if col_funcs is None:
            col_funcs = {}

        assert isinstance(col_funcs, dict), f"col_funcs must be a dictionary, got type: {type(col_funcs)}"

        # add columns
        for new_col, col_fun in col_funcs.items():

            # add new column
            if verbose >= 3:
                print(f"adding new_col: {new_col}")
            df[new_col] = config_func(df=df, **col_fun)

        # --
        # col checks
        # --

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

        da_list = []
        for idx, bcp in bc_pair.iterrows():

            # select data
            select = np.ones(len(df), dtype=bool)
            for bc in by_cols:
                select &= (df[bc] == bcp[bc]).values
            df_bin = df.loc[select, :]

            # store the 'by' coords
            by_coords = {bc: [bcp[bc]] for bc in by_cols}

            b, xc, yc = cls.bin_data(df_bin,
                                     x_range=x_range,
                                     y_range=y_range,
                                     grid_res=grid_res,
                                     x_col=x_col,
                                     y_col=y_col,
                                     val_col=val_col,
                                     bin_statistic=bin_statistic,
                                     return_bin_center=True)
            # add extra dimensions to binned data
            b = b.reshape(b.shape + (1,) * len(by_cols))
            # store data in DataArray
            # TODO: review y,x order - here assumes y is first dim. for symmetrical grids it doesn't matter
            coords = {**{'y': yc, 'x': xc}, **by_coords}
            da = xr.DataArray(data=b,
                              dims=['y', 'x'] + by_cols,
                              coords=coords,
                              name=val_col)
            da_list += [da]

        # combine into a single Dataset
        out = xr.combine_by_coords(da_list)
        return out

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
                 return_bin_center=True):
        """

        Parameters
        ----------
        df
        x_range
        y_range
        grid_res
        x_col
        y_col
        val_col
        bin_statistic
        return_bin_center

        Returns
        -------

        """
        # TODO: complete doc string
        # TODO: move defaults out of bin_data to bin_data_by?
        # TODO: double check get desired shape, dim alignment if x_range != y_range

        # ---
        # check inputs, handle defaults

        assert val_col is not None, "val_col - the column containing values to bin cannot be None"
        assert grid_res is not None, "grid_res is None, must be supplied - expressed in km"
        assert len(df) > 0, f"dataframe (df) provide must have len > 0"

        if x_range is None:
            x_range = [-4500000.0, 4500000.0]
            print(f"x_range, not provided, using default: {x_range}")
        assert x_range[0] < x_range[1], f"x_range should be (min, max), got: {x_range}"

        if y_range is None:
            y_range = [-4500000.0, 4500000.0]
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
        binned = scst.binned_statistic_2d(x_in, y_in, vals,
                                          statistic=bin_statistic,
                                          bins=[x_edge,
                                                y_edge],
                                          range=[[x_min, x_max], [y_min, y_max]])

        xy_out = x_edge, y_edge
        # return the bin centers, instead of the edges?
        if return_bin_center:
            x_cntr, y_cntr = x_edge[:-1] + np.diff(x_edge) / 2, y_edge[:-1] + np.diff(y_edge) / 2
            xy_out = x_cntr, y_cntr

        # TODO: if output is transpose, should the x,y (edges or centers) be swapped?
        return binned[0].T, xy_out[0], xy_out[1]


if __name__ == "__main__":

    pass
