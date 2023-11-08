# class for creating prediction locations
# - e.g. relative an expert location

import numpy as np
import numba as nb
import pandas as pd

from typing import List, Dict, Tuple, Union, Type
from GPSat.utils import to_array, match
from GPSat.decorators import timer
from GPSat.dataloader import DataLoader

# ---
# helper function
# ---

# @timer
@nb.guvectorize([(nb.float64[:, :], nb.float64[:], nb.float64[:], nb.bool_[:])],
                '(n, d),(d), () -> (n)')
def _max_dist_bool(loc, ref_loc, max_dist, out):
    """given location (loc) array - shape (n,d) - and
    reference / expert location (ref_loc) - shape (d,) - and
    maximum distance value (max_dist)
    return bool array indication which locations are further than max_dist from ref_loc using p2 norm

    motivation was to run quickly for ~100M locations per expert location
    """
    # initialise all the output to be True
    out[:] = True

    # for each dimension loc determine those too far away (e.g. l1 norm)
    for j in range(loc.shape[1]):
        for i in range(loc.shape[0]):
            if out[i]:
                d = loc[i, j] - ref_loc[j]
                # out[i] = np.abs(d) < max_dist[0]
                out[i] = (d*d) < (max_dist[0] * max_dist[0])

    # out[out] = np.sum( (loc[out, :] - ref_loc) ** 2)  < (max_dist[0] * max_dist[0])
    for i in range(loc.shape[0]):
        if out[i]:
            d2 = np.sum( (loc[i, :] - ref_loc) ** 2)
            out[i] = d2 < (max_dist[0] * max_dist[0])


# ---
# class def
# ---

class PredictionLocations:

    # TODO: double check the data types
    _coords_col: Union[list, None] = None
    _exprt_loc: Union[np.ndarray, pd.DataFrame, None] = None
    # _kwargs: Union[dict, None] = None

    def __init__(self,
                 method="expert_loc",
                 coords_col=None,
                 expert_loc=None,
                 **kwargs):
        self.method = method

        # keyword arguments to provided to selected method for generating prediction locations
        self.kwargs = kwargs

        self._coords_col = coords_col
        self.expert_loc = expert_loc
        # TODO: check is value name - prediction location approach
        # TODO: should kwargs be assigned as attribute?

    def __call__(self):
        # prediction locations
        # allow for instance of class to be called and return array of prediction locations

        if self.method == "shift_arrays":
            out = self._shift_arrays(**self.kwargs)
        elif self.method == "expert_loc":
            out = self.expert_loc
        elif self.method == "from_dataframe":
            out = self._from_dataframe(**self.kwargs)
        elif self.method == "from_source":
            # from_source allows use of DataLoader.load to initially load a DataFrame
            # - after which (subsequent calls) method will be set to from_dataframe,
            # - with the loaded DataFrame being stored in kwargs
            assert "load_kwargs" in self.kwargs, \
                "calling PredictionLocations object with method='from_source', however 'load_kwargs' is missing from" \
                " kwargs, these arguments are to be provided to DataLoader.load"

            load_kwargs = self.kwargs.pop("load_kwargs")
            df = DataLoader.load(**load_kwargs)
            # drop duplicates, just in case
            df = df.drop_duplicates()

            # TODO: review this - as it stands this would only allow for loading one set of prediction locations
            # update the method and put df in kwargs for future use
            self.method = "from_dataframe"
            self.kwargs['df'] = df

            out = self._from_dataframe(**self.kwargs)

        else:
            raise ValueError(f"name: '{self.name}' not implement")

        # apply local select? e.g. select prediction locations that have same dim, say t, as reference/expert location
        if (self.method == "from_dataframe") & ("local_select" in self.kwargs):

            out = DataLoader.local_data_select(out,
                                               reference_location=self.expert_loc,
                                               local_select=self.kwargs["local_select"],
                                               verbose=False)

        assert isinstance(out, np.ndarray), f"must return ndarray, got: {type(out)}"
        assert len(out.shape) == 2, f"must return 2d array, got len {len(out.shape)}d"
        return out

    @property
    def coords_col(self):
        # using @property to use setter functionality
        return self._coords_col

    @coords_col.setter
    def coords_col(self, value):
        # TODO: apply checks on value - make sure they are set as list (?) or is None
        if value is None:
            self._coords_col = value
        elif isinstance(value, np.ndarray):
            # require coords_col be a 1d array
            assert len(value.shape) == 1
            self._coords_col = value.tolist()
        elif isinstance(value, list):
            self._coords_col = value
        else:
            raise ValueError(f"in setting coords_col - value is type: {type(value)} is not handled")

    @property
    def expert_loc(self):
        # should this be a copy?
        return self._expert_loc

    @staticmethod
    def _1row_2d_array(x):

        # if list provided convert to array
        if isinstance(x, list):
            x = np.array(x)
        assert isinstance(x, np.ndarray)
        # if 1d, broadcast to 2d
        if len(x.shape) == 1:
            x = x[None, :]

        if len(x.shape) == 2:
            # require only have 1 row
            assert x.shape[0] == 1
        else:
            raise NotImplementedError(f"array provided has dim larger than 2d - got: {len(x.shape)}")

        return x

    @expert_loc.setter
    def expert_loc(self, value):
        # set expert location - as 2d numpy array with 1 row
        # and columns in corresponding to coords_col
        if isinstance(value, np.ndarray):
            self._expert_loc = self._1row_2d_array(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            assert self.coords_col is not None, f"in setting expert_loc was provide {type(value)}, " \
                                                f"however coords_col is None - it must be set"
            # select the coords_col from DataFrame / Series - check / make into 1 row 2d array
            self._expert_loc = self._1row_2d_array(value[self.coords_col].values)
        elif isinstance(value, list):
            self._expert_loc = self._1row_2d_array(value)
        elif value is None:
            self._expert_loc = None
        else:
            raise ValueError(f"in setting expert_loc - value is type: {type(value)} is not handled")

    def _to_array(self, x):
        out, = to_array(x)
        return out

    def _shift_arrays(self, Xout=None, **kwargs):
        # TODO: rename this to mesh grid or something

        if Xout is None:
            # get shift arrays for each coord - default to empty zeros array
            xis = [self._to_array(kwargs.get(c, np.zeros(1))) for c in self.coords_col]
            for x in xis:
                assert len(x.shape) == 1

            # use meshgrid to get all combinations of coordinates, flatten and concat
            Xis = np.meshgrid(*xis, indexing='ij')
            Xis = [X.flatten()[:, None] for X in Xis]
            Xout = np.concatenate(Xis, axis=1)
            # store Xout in kwargs for future use
            self.kwargs['Xout'] = Xout

        # expert location
        xel = self.expert_loc

        # appy a shift to the expert location
        # TODO: apply dimension check on Xout
        Xout = Xout + xel

        return Xout

    # @timer
    def _from_dataframe(self, df=None, df_file=None, max_dist=None, copy_df=False, **kwargs):
        # TODO: rename df_file
        # TODO: finish this method - should out array be created each time?
        # TODO: review this method... it's a mess...
        # TODO: properly implement or remove copy_df - leaning towards remove
        if df is None:

            assert isinstance(df_file, (str, dict)), f"df is None, df_file expected to be str or dict, got: {type(df_file)}"
            #
            if isinstance(df_file, str):
                df = pd.read_csv(df_file)
            elif isinstance(df_file, str):
                pass

            # find the columns in dataframe that are in coords_col
            # found_cols = [c for c in df.columns if c in self.coords_col]
            found_cols = [c for c in self.coords_col if c in df.columns]

            # select only the relevant data
            df = df.loc[:, found_cols]
            # store df in kwargs so don't have to re-read in (will be passed in next time)
            self.kwargs['df'] = df.copy(True) if copy_df else df
        else:
            # find the columns in dataframe that are in coords_col
            found_cols = [c for c in self.coords_col if c in df.columns]

            # TODO: check if df columns are entirely in self.coords_col, if not drop extra, store in kwargs?
            # remove any extra, unneeded columns
            if df.shape[1] > len(found_cols):
                df = df.loc[:, found_cols]
                # if taking reduced set of columns, store for next time
                self.kwargs['df'] = df.copy(True) if copy_df else df

        # for each of the found_cols get the location in self.coords_col
        # - for selecting correct expert location columns
        fc_loc = [match(c, self.coords_col)[0] for c in found_cols]

        # check for max dist - get a bool selection array
        if max_dist is not None:
            # create an array to get re-used - will be helpful if b ~ 100M rows
            # HACK: if, for some reason, the dtypes don't match, cast the expert location as same time as data
            if self.expert_loc.dtype != df.values.dtype:
                self.expert_loc = self.expert_loc.astype(df.values.dtype)
            b = self._max_dist_bool(df.values,
                                    self.expert_loc[:, fc_loc],
                                    max_dist)
        # other wise use slice(None) - to select all rows of DataFrame
        else:
            b = slice(None)

        # populate output array
        # if found all columns just return values
        if len(found_cols) == len(self.coords_col):
            out = df.loc[b, :].values
        else:
            # create an array to fill
            nrow_out = len(df) if isinstance(b, slice) else b.sum()
            out = np.full((nrow_out, len(self.coords_col)), np.nan)
            out[:, fc_loc] = df.loc[b, :].values

            # for the missing dimension populate with expert location value
            missing_cols = [cc for cc in self.coords_col if cc not in found_cols]
            missing_col_loc = match(missing_cols, self.coords_col)
            out[:, missing_col_loc] = self.expert_loc[:, missing_col_loc]

        return out


    # @timer
    def _max_dist_bool(self, locs, exp_loc, max_dist):
        # TODO: shape checks
        # return np.sqrt(np.sum( (locs - exp_loc)**2, axis=1)) <= max_dist
        # return np.sum((locs - exp_loc) ** 2, axis=1, out=out) <= (max_dist * max_dist)
        return _max_dist_bool(locs, exp_loc[0, :], max_dist)


if __name__ == "__main__":

    from scipy.spatial.distance import cdist
    from GPSat.utils import grid_2d_flatten

    # parts of the below could be used for unittests
    # ---
    # default method - predict at expert location only
    # ---

    coords_col = ['x', 'y', 't']
    xprt_loc = pd.DataFrame({'x': 0., 'y': 2., "t": 3.}, index=[0])

    ploc = PredictionLocations(coords_col=coords_col,
                               expert_loc=xprt_loc,
                               **{})

    # check expert_loc property is correct type
    assert isinstance(ploc.expert_loc, np.ndarray)
    # and when using __call__ get desired out
    assert np.all(ploc() == ploc.expert_loc)

    # --
    # setting  and getting expert location
    # --

    # w pandas DataFrame
    # NOTE: when setting values convert to 2d array
    print(type(xprt_loc))
    ploc.expert_loc = xprt_loc
    el0 = ploc.expert_loc

    # pandas Series
    print(type(xprt_loc.iloc[0, :]))
    ploc.expert_loc = xprt_loc.iloc[0, :]
    el1 = ploc.expert_loc

    # numpy array
    ploc.expert_loc = xprt_loc.iloc[0, :].values
    el2 = ploc.expert_loc

    # with a list
    ploc.expert_loc = xprt_loc.iloc[0, :].values.tolist()
    el3 = ploc.expert_loc

    # the effective results should be the same: 2d numpy array
    assert len(el0.shape) == 2
    np.testing.assert_array_equal(el0, el1)
    np.testing.assert_array_equal(el0, el2)
    np.testing.assert_array_equal(el0, el3)

    # the default method='expert_loc' should just return the expert location
    np.testing.assert_array_equal(el0, ploc())

    # ----
    # use shift_arrays - using meshgrid to create locations shifted from expert location
    # ----

    # define inputs via a dict
    shifts = {
        "x": np.arange(3),
        "y": np.linspace(-2, 3, 4)
    }

    pred_loc_params = {
        "method": "shift_arrays",
        **shifts
    }

    ploc = PredictionLocations(**pred_loc_params)

    # set coords_col and expert location
    ploc.coords_col = coords_col
    ploc.expert_loc = xprt_loc

    pred_locs = ploc()
    pred_locs2 = ploc()

    # confirm the number of prediction locations is correct
    # correct shape
    assert len(pred_locs.shape) == 2
    # columns have same length as coords_col
    assert pred_locs.shape[1] == len(ploc.coords_col)
    # number of rows should be product of array dimension
    assert pred_locs.shape[0] == np.prod([len(v) for v in shifts.values()])

    # ----
    # from_dataframe - fixed locations
    # ----

    # generate a large 'fine grain' grid - every 1000m - creates 81M rows
    xy_range = [-4500000.0, 4500000.0]
    X = grid_2d_flatten(xy_range, xy_range, step_size=5 * 1000)
    df = pd.DataFrame(X, columns=['y', 'x'])

    # provide dataframe directly
    max_dist = 100 * 1000
    ploc = PredictionLocations(method="from_dataframe",
                               df=df,
                               max_dist=max_dist)
    ploc.coords_col = coords_col
    ploc.expert_loc = xprt_loc

    pred_loc = ploc()

    # validate maximum distance is capped
    r = cdist(pred_loc, ploc.expert_loc)
    assert r.max() < max_dist

    # specify a t column - same as before
    df['t'] = 3.0
    ploc = PredictionLocations(method="from_dataframe",
                               df=df,
                               max_dist=max_dist,
                               coords_col=coords_col,
                               expert_loc=xprt_loc)

    pred_loc2 = ploc()
    np.testing.assert_array_equal(pred_loc, pred_loc2)

    # move 't' outside of max_dist - so will choose None
    df['t'] = max_dist + 1e-6
    ploc2 = PredictionLocations(method="from_dataframe",
                                df=df,
                                max_dist=max_dist,

                                coords_col=coords_col,
                                expert_loc=xprt_loc)

    # should return zero row array
    assert len(ploc2()) == 0




