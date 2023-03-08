# class for creating prediction locations
# - e.g. relative an expert location

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Type
from PyOptimalInterpolation.utils import to_array


class PredictionLocations:

    # TODO: double check the data types
    _coords_col: Union[list, None] = None
    _exprt_loc: Union[np.ndarray, pd.DataFrame, None] = None

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
        # allow for instance of class to be called and return array of prediction locations

        if self.method == "shift_arrays":
            out = self.shift_arrays(**self.kwargs)
        elif self.method == "expert_loc":
            out = self.expert_loc
        else:
            raise ValueError(f"name: '{self.name}' not implement")

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

    def shift_arrays(self, **kwargs):
        # TODO: rename this to mesh grid or something

        # get shift arrays for each coord - default to empty zeros array
        xis = [self._to_array(kwargs.get(c, np.zeros(1))) for c in self.coords_col]
        for x in xis:
            assert len(x.shape) == 1

        # use meshgrid to get all combinations of coordinates, flatten and concat
        Xis = np.meshgrid(*xis, indexing='ij')
        Xis = [X.flatten()[:, None] for X in Xis]
        Xout = np.concatenate(Xis, axis=1)

        # expert location
        xel = self.expert_loc

        # appy a shift to the expert location
        # TODO: apply dimension check on Xout
        Xout = Xout + xel

        return Xout


if __name__ == "__main__":

    # parts of the below could be used for unittests
    # ---
    # default method - predict at expert location only
    # ---

    coords_col = ['x', 'y', 't']
    xprt_loc = pd.DataFrame({'x': 0, 'y': 2, "t": 3}, index=[0])

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

    # confirm the number of prediction locations is correct
    # correct shape
    assert len(pred_locs.shape) == 2
    # columns have same length as coords_col
    assert pred_locs.shape[1] == len(ploc.coords_col)
    # number of rows should be product of array dimension
    assert pred_locs.shape[0] == np.prod([len(v) for v in shifts.values()])

