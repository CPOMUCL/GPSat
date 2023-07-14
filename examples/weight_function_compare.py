# playground for weight functions


import numba as nb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.convolution import convolve, Gaussian2DKernel

from GPSat.decorators import timer

@timer
@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
                 (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
                  nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:])],
                '(), (), (n), (n), (), (), (n)->()',
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

@timer
def convolve_wrapper(z, l_x=1, l_y=1):
    return  convolve(z, Gaussian2DKernel(x_stddev=l_x, y_stddev=l_y))

# ----
# create some random data
# ----

n = 10000
df = pd.DataFrame({'x': np.random.normal(size=n),
                   'y': np.random.normal(size=n),
                   'z': np.random.normal(size=n)})

coords_col = ['x', 'y']
val_col = 'z'


x0, y0 = [df[_].values for _ in coords_col]
x, y = [df[_].values for _ in coords_col]
vals = df[val_col].values

_ = gaussian_2d_weight(x0, y0, x, y, 1, 1, vals)


# ---
# validate with convolve
# ---

n, m = 20, 20
x = np.arange(n, dtype=float)
y = np.arange(m, dtype=float)
x_grid, y_grid = np.meshgrid(x, y)
# z = np.random.normal(size=(n, m))
# z = np.zeros((n,m))
z = np.ones((n, m))

# put a line across the middle
# z[n//2,:] = 1.0

# fil some values with nan
# - lines
z[n//2, :] = np.nan
z[:, m//2] = np.nan
# - block
z[n//3:(n-n//3), :m//3] = np.nan
# z[:, m//3:(m-m//3)] = np.nan
# z[n//2, m//2] = np.nan

# randomly assign nans

df = pd.DataFrame({"x": x_grid.flatten(), "y": y_grid.flatten(), "z": z.flatten()})

# reference output
# ref = convolve(z, Gaussian2DKernel(x_stddev=1, y_stddev=1))
ref = convolve_wrapper(z, l_x=1, l_y=1)

#
x0, y0 = [df[_].values for _ in coords_col]
x, y = [df[_].values for _ in coords_col]
vals = df[val_col].values

_ = gaussian_2d_weight(x0, y0, x, y, 1, 1, vals)

# ---
# plot results
# ---


fig = plt.figure(figsize=(15, 5))
fig.suptitle("Showing convolve w Gaussian2DKernel can bias towards zero\non edges and near nans")

vmin, vmax = np.nanmin(ref), np.nanmax(ref)

ax = fig.add_subplot(1, 3, 1)
s = ax.imshow(z,
          vmin=vmin, vmax=vmax)
ax.set_title("original data - white: nan")
fig.colorbar(s,  orientation='vertical')

ax = fig.add_subplot(1, 3, 2)
s = ax.imshow(ref,
          vmin=vmin, vmax=vmax)
ax.set_title("smooth with convolve Gaussian2DKernel")
fig.colorbar(s,  orientation='vertical')

ax = fig.add_subplot(1, 3, 3)
tmp = ax.imshow(_.reshape(n, m),
          vmin=vmin, vmax=vmax)
ax.set_title("smooth with custom weight function")

# colorbar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(tmp, cax=cax, orientation='vertical')
fig.colorbar(tmp,  orientation='vertical')
plt.tight_layout()

plt.show()


# ---
# compare smoothed results directly

plt.imshow(_.reshape(n, m) - ref,
           cmap="bwr")
plt.colorbar()
plt.title("differences of weighted averages")
plt.show()



