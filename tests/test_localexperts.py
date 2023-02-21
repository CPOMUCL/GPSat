#%%
# Testing local experts
import numpy as np
import pandas as pd
import gpflow
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from PyOptimalInterpolation.models import GPflowGPRModel, GPflowSGPRModel, sklearnGPRModel
from PyOptimalInterpolation.models.vff_model import GPflowVFFModel

# Generate random data from matern-3/2 model
np.random.seed(23435)

kernel = Matern(length_scale=0.8, nu=3/2)
gp = GaussianProcessRegressor(kernel)

x = np.linspace(0, 10, 100)[:,None]
f = gp.sample_y(x, random_state=0)

N = 50
eps = 1e-2
indices = np.arange(100)
np.random.shuffle(indices)
x_train = x[indices[:N]]
y_train = f[indices[:N]] + eps*np.random.randn(N,1)

df = pd.DataFrame(data={'x': x_train[:,0], 'y': y_train[:,0]})

# Fit matern-3/2 gp on training data
gp.alpha = eps**2
gp.fit(x_train, y_train)
ls = gp.kernel_.length_scale
ml = gp.log_marginal_likelihood()

# Get prediction at random point
test_index = np.random.randint(0,99)
x_test = x[[test_index]]
pred_mean, pred_std = gp.predict(x_test, return_std=True)

#%%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# model = GPflowVFFModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None,
#                         num_inducing_features=40)

# low=1e-10; high=1e5
# model.set_lengthscale_constraints(low=low, high=high)

# model.model.elbo()

# #%%
# model.set_parameters(likelihood_variance=eps**2)
# gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
# gpflow.set_trainable(model.model.kernel.variance, False)
# model.optimise_parameters()

#%%

class TestLocalExperts:
    def test_gpflow_gpr(self, tol=1e-7, low=1e-10, high=1e5):
        model = GPflowGPRModel(data=df,
                               obs_col='y',
                               coords_col='x',
                               obs_mean=None)

        model.set_parameters(likelihood_variance=eps**2)
        gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
        gpflow.set_trainable(model.model.kernel.variance, False)

        model.set_lengthscale_constraints(low=low, high=high)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)

        assert np.abs(result['marginal_loglikelihood'] - ml) < tol
        assert np.abs(result['lengthscales'] - ls) < tol
        assert np.abs(out['f*'] - pred_mean) < tol
        assert np.abs(out['f*_var'] - pred_std**2) < tol

    def test_gpflow_sgpr(self, tol=1e-4, low=1e-10, high=1e5):
        model = GPflowSGPRModel(data=df,
                                obs_col='y',
                                coords_col='x',
                                obs_mean=None,
                                num_inducing_points=None)

        model.set_parameters(likelihood_variance=eps**2)
        gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
        gpflow.set_trainable(model.model.kernel.variance, False)

        model.set_lengthscale_constraints(low=low, high=high)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)

        # assert np.abs(result['marginal_loglikelihood'] - ml) < tol
        assert np.abs(result['lengthscales'] - ls) < tol
        assert np.abs(out['f*'] - pred_mean) < tol
        assert np.abs(out['f*_var'] - pred_std**2) < tol

    def test_gpflow_vff(self, tol=1e-4, low=1e-10, high=1e5):
        model = GPflowVFFModel(data=df,
                               obs_col='y',
                               coords_col='x',
                               obs_mean=None,
                               num_inducing_features=25,
                               margin=[1.0])

        model.set_parameters(likelihood_variance=eps**2)
        gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
        gpflow.set_trainable(model.model.kernel.variance, False)

        model.set_lengthscale_constraints(low=low, high=high)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)

        # assert np.abs(result['marginal_loglikelihood'] - ml) < tol
        # assert np.abs(result['lengthscales'] - ls) < tol
        # assert np.abs(out['f*'] - pred_mean) < tol
        # assert np.abs(out['f*_var'] - pred_std**2) < tol

    def test_scikit(self, tol=1e-7, low=1e-10, high=1e5):
        model = sklearnGPRModel(data=df,
                                obs_col='y',
                                coords_col='x',
                                obs_mean=None,
                                likelihood_variance=eps**2)

        model.set_lengthscale_constraints(low=low, high=high)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)

        assert np.abs(result['marginal_loglikelihood'] - ml) < tol
        assert np.abs(result['lengthscales'] - ls) < tol
        assert np.abs(out['f*'] - pred_mean) < tol
        assert np.abs(out['f*_var'] - pred_std**2) < tol



# %%
