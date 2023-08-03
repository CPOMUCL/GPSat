#%%
# Testing local experts
import numpy as np
import pandas as pd
import gpflow
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
#from GPSat.models import GPflowGPRModel, GPflowSGPRModel, GPflowSVGPModel, sklearnGPRModel
from GPSat.models import get_model
# from GPSat.models.vff_model import GPflowVFFModel
# from GPSat.models.gpytorch_models import GPyTorchGPRModel
# from GPSat.models.asvgp_model import GPflowASVGPModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable GPU

# get the models
GPflowGPRModel, GPflowSGPRModel, GPflowSVGPModel, \
    sklearnGPRModel, GPflowVFFModel, GPyTorchGPRModel = \
    [get_model(m) for m in ['GPflowGPRModel', 'GPflowSGPRModel', 'GPflowSVGPModel',
                            'sklearnGPRModel', 'GPflowVFFModel', 'GPyTorchGPRModel']]

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


constraints_dict = {
            'lengthscales': {'low': 1e-10, 'high': 5.}
        }

# #%%
# model = GPyTorchGPRModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None,
#                         noise_variance=eps**2)

# model.set_parameter_constraints(constraints_dict)

# result = model.optimise_parameters()
# out = model.predict(coords=x_test)


# #%%
# model = sklearnGPRModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None,
#                         likelihood_variance=eps**2)
# constraints_dict = {
#     'lengthscales': {'low': 1e-10, 'high': 0.9},
#     'kernel_variance': {'low': 1e-10, 'high': 2.}
# }
# model.set_parameter_constraints(constraints_dict)

# result = model.optimise_parameters()
# out = model.predict(coords=x_test)

# %%
# model = GPflowGPRModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None)
# model.set_parameters(likelihood_variance=eps**2)
# gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
# gpflow.set_trainable(model.model.kernel.variance, False)
# constraints_dict = {
#     'lengthscales': {'low': 1e-10, 'high': 5.},
# }
# model.set_parameter_constraints(constraints_dict)

# model.get_objective_function_value()

# model.optimise_parameters()

# result = model.optimise_parameters()
# out = model.predict(coords=x_test)

#%%
# model = GPflowSGPRModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None,
#                         num_inducing_points=None)

# constraints_dict = {
#     'lengthscales': {'low': 1e-10, 'high': 0.9},
#     'kernel_variance': {'low': 1e-10, 'high': 1.},
#     'likelihood_variance': {'low': 1e-10, 'high': 0.1},
# }
# # model.set_parameters(likelihood_variance=eps**2)
# model.set_parameter_constraints(constraints_dict)

# model.get_objective_function_value()

# # model.model.elbo()

# model.set_parameters(likelihood_variance=eps**2)
# gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
# gpflow.set_trainable(model.model.kernel.variance, False)

# model.optimise_parameters()

#%%
# model = GPflowSVGPModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None,
#                         num_inducing_points=None,
#                         minibatch_size=None)

# constraints_dict = {
#     'lengthscales': {'low': 1e-10, 'high': 0.9},
#     'kernel_variance': {'low': 1e-10, 'high': 1.},
#     'likelihood_variance': {'low': 1e-10, 'high': 0.1},
# }
# # model.set_parameters(likelihood_variance=eps**2)
# model.set_parameter_constraints(constraints_dict)

# model.get_objective_function_value()

# # model.model.elbo()

# model.set_parameters(likelihood_variance=eps**2)
# # gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
# # gpflow.set_trainable(model.model.kernel.variance, False)

# model.optimise_parameters(learning_rate=1e-2, gamma=0.1, verbose=True)

#%%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# model = GPflowVFFModel(data=df,
#                         obs_col='y',
#                         coords_col='x',
#                         obs_mean=None,
#                         num_inducing_features=40)

# constraints_dict = {
#     'lengthscales': {'low': 1e-10, 'high': 0.9},
#     'kernel_variance': {'low': 1e-10, 'high': 2.},
#     'likelihood_variance': {'low': 1e-10, 'high': 0.1},
# }
# model.set_parameters(likelihood_variance=eps**2)
# model.set_parameter_constraints(constraints_dict)

# model.model.elbo()

# model.set_parameters(likelihood_variance=eps**2)
# gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
# gpflow.set_trainable(model.model.kernel.variance, False)
# model.optimise_parameters()

# #%%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# model = GPflowASVGPModel(data=df,
#                          obs_col='y',
#                          coords_col='x',
#                          obs_mean=None,
#                          num_inducing_features=40)

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
    def test_gpflow_gpr(self, tol=1e-6):
        model = GPflowGPRModel(data=df,
                               obs_col='y',
                               coords_col='x',
                               obs_mean=None)

        model.set_parameters(likelihood_variance=eps**2)
        gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
        gpflow.set_trainable(model.model.kernel.variance, False)

        model.set_parameter_constraints(constraints_dict)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)
        params = model.get_parameters()
        # objective function is negative log marginal likelihood
        # - take negative to get ml
        objfunc = -model.get_objective_function_value()
        # optimisation should have succeeded
        assert result
        assert np.abs(params['lengthscales'][0] - ls) < tol
        assert np.abs(objfunc - ml) < tol
        assert np.abs(out['f*'] - pred_mean) < tol
        assert np.abs(out['f*_var'] - pred_std**2) < tol

    def test_gpflow_sgpr(self, tol=1e-4):
        model = GPflowSGPRModel(data=df,
                                obs_col='y',
                                coords_col='x',
                                obs_mean=None,
                                num_inducing_points=50)

        model.set_parameters(likelihood_variance=eps**2)
        gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
        gpflow.set_trainable(model.model.kernel.variance, False)

        model.set_parameter_constraints(constraints_dict)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)
        params = model.get_parameters()

        assert result
        assert np.abs(params['lengthscales'][0] - ls) < tol
        # assert np.abs(result['marginal_loglikelihood'] - ml) < tol
        # assert np.abs(result['lengthscales'] - ls) < tol
        assert np.abs(out['f*'] - pred_mean) < tol
        assert np.abs(out['f*_var'] - pred_std**2) < tol

    # def test_gpflow_vff(self):
    #     # TODO: complete this test
    #     model = GPflowVFFModel(data=df,
    #                            obs_col='y',
    #                            coords_col='x',
    #                            obs_mean=None,
    #                            num_inducing_features=25,
    #                            margin=[1.0])

    #     model.set_parameters(likelihood_variance=eps**2)
    #     gpflow.set_trainable(model.model.likelihood.variance, False) # TODO: Write as method
    #     gpflow.set_trainable(model.model.kernel.variance, False)

    #     model.set_parameter_constraints(constraints_dict)

    #     result = model.optimise_parameters()
    #     out = model.predict(coords=x_test)

    #     # assert np.abs(result['marginal_loglikelihood'] - ml) < tol
    #     # assert np.abs(result['lengthscales'] - ls) < tol
    #     # assert np.abs(out['f*'] - pred_mean) < tol
    #     # assert np.abs(out['f*_var'] - pred_std**2) < tol

    def test_scikit(self, tol=1e-1):
        model = sklearnGPRModel(data=df,
                                obs_col='y',
                                coords_col='x',
                                obs_mean=None,
                                kernel_variance=None, # Sets kv to 1 and does not get trained
                                likelihood_variance=eps**2)

        model.set_parameter_constraints(constraints_dict)

        result = model.optimise_parameters()
        out = model.predict(coords=x_test)
        params = model.get_parameters()
        objfunc = model.get_objective_function_value()

        assert result
        assert np.abs(params['lengthscales'] - ls) < tol
        assert np.abs(objfunc - ml) < tol
        assert np.abs(out['f*'] - pred_mean) < tol
        assert np.abs(out['f*_var'] - pred_std**2) < tol

    # def test_gpytorch(self, tol=1e-7):
    #     model = GPyTorchGPRModel(data=df,
    #                             obs_col='y',
    #                             coords_col='x',
    #                             obs_mean=None,
    #                             noise_variance=eps**2)

    #     model.set_parameter_constraints(constraints_dict)

    #     result = model.optimise_parameters()
    #     out = model.predict(coords=x_test)

    #     params = model.get_parameters()
    #     objfunc = model.get_objective_function_value()

    #     assert result
    #     assert np.abs(params['lengthscales'] - ls) < tol
    #     assert np.abs(objfunc - ml) < tol
    #     assert np.abs(out['f*'] - pred_mean) < tol
    #     assert np.abs(out['f*_var'] - pred_std**2) < tol



# %%
