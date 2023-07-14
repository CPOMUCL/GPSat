# module for a "Pure Python" implementation of GPR using a Matern 3/2 Kernel
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot as mdot
import scipy
from scipy.spatial.distance import squareform, pdist, cdist

from GPSat.decorators import timer
from GPSat.models import BaseGPRModel

from GPSat.utils import sigmoid, inverse_sigmoid, softplus, inverse_softplus

class PurePythonGPR(BaseGPRModel):
    """Pure Python GPR class - used to hold model details from pure python implementation"""

    def __init__(self,
                 data=None,
                 coords_col=None,
                 obs_col=None,
                 coords=None,
                 obs=None,
                 coords_scale=None,
                 obs_scale=None,
                 obs_mean=None,
                 *,
                 length_scales=1.0,
                 kernel_var=1.0,
                 likeli_var=1.0,
                 kernel="Matern32",
                 constraints_dict=None,
                 **kwargs):
        assert kernel == "Matern32", "only 'Matern32' kernel handled"

        # TODO: check values, make sure hyper parameters can be concatenated together
        # --
        # set data
        # --

        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale,
                         obs_mean=obs_mean)

        # just store a values as attributes
        self.x = self.coords
        self.y = self.obs

        # TODO: perform shape check

        self.set_lengthscales(lengthscales=length_scales)
        self.set_kernel_variance(kernel_var)
        self.set_likelihood_variance(likeli_var)

        # ---
        # hyper parameters constraint / transform functions
        # ---

        # the transform function will be stored in a dictionary
        # - key: parameter name
        # - value dict containing:  func, inverse_func, kwargs
        # - OR value list containing dicts with keys: func, inverse_func, kwargs

        self.transforms = {}

        if constraints_dict is None:
            constraints_dict = {}

        for k in self.param_names:
            if k not in constraints_dict:
                constraints_dict[k] = {"func": "softplus"}

            if "func" not in constraints_dict[k]:
                print(f"guessing constraint function for: {k}")
                constraints_dict[k]["func"] = self._guess_constraint_func(**constraints_dict[k])

        # set the transform function
        self.set_parameter_constraints(constraints_dict)

    def _guess_constraint_func(self, **kwargs):

        if ("low" in kwargs) & ("high" in kwargs):
            return "sigmoid"
        elif "shift" in kwargs:
            return "softplus"
        else:
            warnings.warn("unable to guess constraint / transform function based on additional arguments")
            return "softplus"

    @property
    def param_names(self) -> list:
        return ["lengthscales", "kernel_variance", "likelihood_variance"]

    def get_lengthscales(self):
        return self.length_scales

    def get_kernel_variance(self):
        return self.kernel_var

    def get_likelihood_variance(self):
        return self.likeli_var

    def set_lengthscales(self, lengthscales):

        if isinstance(lengthscales, (int, float)):
            lengthscales = np.full(self.coords.shape[1],
                                   lengthscales,
                                   dtype=float)

        elif isinstance(lengthscales, list):
            lengthscales = np.array(lengthscales)

        # TODO: there should be a shape check here, lengthscales should match coords dim
        assert self.coords.shape[1] == len(lengthscales), "lengthscales must align to dim of coords"
        self.length_scales = lengthscales

    def set_kernel_variance(self, kernel_variance):
        self.kernel_var = kernel_variance

    def set_likelihood_variance(self, likelihood_variance):
        self.likeli_var = likelihood_variance

    def get_transform_funcs(self,  func, **kwargs):
        # assert name in self.param_names, f"name: {name} not in param_names: {self.param_names}"

        if func == "softplus":
            return {"func": softplus, "inv_func": inverse_softplus, "kwargs": kwargs}
        elif func == "exp":
            # kwargs not used for these transform functions, for now
            return {"func": np.exp, "inv_func": np.log, "kwargs": {}}
        elif func == "sigmoid":
            return {"func": sigmoid, "inv_func": inverse_sigmoid, "kwargs": kwargs}
        # TODO: add a linear transform
        else:
            raise NotImplementedError(f"func: {func} is not implement")

    def _move_to_within_bound(self, chk, tol=1e-2, **kwargs):
        # NOTE: this is not robust; expects certain keywords to exists
        # NOTE: if too close to boundary, might get stuck (due to low gradient)
        if np.isinf(chk):
            # out of range, to the left
            if chk < 0:
                if "shift" in kwargs:
                    return kwargs["shift"] + tol
                elif "low" in kwargs:
                    return kwargs["low"] + tol
                else:
                    warnings.warn("chk val is -inf not able to move value within bounds")
            else:
                if "high" in kwargs:
                    return kwargs["high"] - tol
                else:
                    warnings.warn("chk is inf but not able to move value within bounds")
        elif np.isnan(chk):
            warnings.warn("the value being checking is NaN, check the constraint function")

    def set_kernel_variance_constraints(self, func=None,  move_within_tol=True, tol=1e-2, **kwargs):

        if func is None:
            func = self._guess_constraint_func(**kwargs)

        # get the the transform dict (contains functions and kwargs)
        _ = self.get_transform_funcs(func, **kwargs)
        self.transforms["kernel_variance"] = _

        # check if the inverse function returns -inf
        if move_within_tol:
            chk = _['inv_func'](self.get_kernel_variance(), **_['kwargs'])
            new_val = self._move_to_within_bound(chk, tol=tol, **_['kwargs'])
            if new_val is not None:
                self.set_kernel_variance(new_val)

    def set_likelihood_variance_constraints(self, func=None, move_within_tol=True,  tol=1e-2, **kwargs):

        if func is None:
            func = self._guess_constraint_func(**kwargs)

        # get the the transform dict (contains functions and kwargs)
        # NOTE: a low of duplicate code here
        _ = self.get_transform_funcs(func, **kwargs)
        self.transforms["likelihood_variance"] = _

        # check if the inverse function returns -inf
        if move_within_tol:
            chk = _['inv_func'](self.get_likelihood_variance(), **_['kwargs'])
            new_val = self._move_to_within_bound(chk, tol=tol, **_["kwargs"])
            if new_val is not None:
                self.set_likelihood_variance(new_val)

    def set_lengthscales_constraints(self, func=None, move_within_tol=True, tol=1e-2, scale=True, **kwargs):
        # each lengthscale (coordindate) should have it's own transform function

        if func is None:
            func = self._guess_constraint_func(**kwargs)

        func = [func] * self.coords.shape[1] if isinstance(func, str) else func

        assert len(func) == self.coords.shape[1], f"lengthscale constraints: len(func)={len(func)} does not match " \
                                                  f"coord dim: {self.coords.shape[1]}"

        tmp = []
        # get the current lengthscale
        ls = self.get_lengthscales().tolist()
        # if isinstance(ls, np.ndarray):
        #     ls = ls.tolist()

        for idx, f in enumerate(func):
            # NOTE: kwargs expected to only be scalars!
            kwrg = {k: v[idx] if isinstance(v, list) else v for k, v in kwargs.items()}

            if scale:
                for k in ["low", "high", "shift"]:
                    if k in kwrg:
                        kwrg[k] /= self.coords_scale[0,idx]

            _ = self.get_transform_funcs(f, **kwrg)
            tmp.append(_)

            # check the current lengthscale value
            if move_within_tol:
                chk = _['inv_func'](ls[idx], **_['kwargs'])
                new_val = self._move_to_within_bound(chk, tol=tol, **kwrg)
                if new_val is not None:
                    ls[idx] = new_val

        self.transforms["lengthscales"] = tmp
        self.set_lengthscales(ls)

        assert isinstance(self.transforms["lengthscales"], list)
        assert len(self.transforms["lengthscales"]) == self.coords.shape[1]

    @timer
    def predict(self, coords, mean=0, apply_scale=True):
        ell = self.length_scales
        sf2 = self.kernel_var
        sn2 = self.likeli_var

        if apply_scale:
            coords = coords / self.coords_scale

        res = GPR(x=self.x,
                  y=self.y,
                  xs=coords,
                  ell=ell,
                  sf2=sf2,
                  sn2=sn2,
                  mean=mean,
                  approx=False,
                  M=None,
                  returnprior=True)

        # TODO: need to confirm these
        # TODO: confirm res[0], etc, can be vectors
        # TODO: allow for mean to be vector

        out = {
            "f*": res[0].flatten(),
            "f*_var": res[1] ** 2,
            # "y": res[0].flatten(),
            "y_var": res[1] ** 2 + self.likeli_var
        }
        return out

    @timer
    def optimise_parameters(self, opt_method="L-BFGS-B", jac=False):
        """an inheriting class should define method for optimising (hyper/variational) parameters"""
        # NOTE: previous default params were: opt_method="CG", jac=True
        return self.optimise(opt_method=opt_method, jac=jac)

    def _apply_transform_funct(self, x0, func_type="func"):
        # apply transform function to concated array of parameters / variables
        # order assumed to be: lengthscale, kernel_variance, likelihood_variance
        # aim to convert parameters to variables and vice versa
        # for variables to params use func_type: "func"
        # for params to variables use func_type: "inv_func"

        assert func_type in ["func", "inv_func"], f"func_type: {func_type} is not valid"

        out = np.full(x0.shape, np.nan)

        # length scales
        num_dim = self.coords.shape[1]
        for i in range(len(self.length_scales)):
            inv_func = self.transforms['lengthscales'][i][func_type]
            kwrgs = self.transforms['lengthscales'][i].get("kwargs", {})
            out[i] = inv_func(x0[i], **kwrgs)

        # kernel variance
        kwrgs = self.transforms['kernel_variance'].get("kwargs", {})
        out[num_dim] = self.transforms['kernel_variance'][func_type](x0[num_dim], **kwrgs)

        # likelihood variance
        kwrgs = self.transforms['likelihood_variance'].get("kwargs", {})
        out[num_dim+1] = self.transforms['likelihood_variance'][func_type](x0[num_dim+1], **kwrgs)

        return out

    def optimise(self, opt_method="L-BFGS-B", jac=False):

        # get the kernel and likelihood variances
        kv = np.array([self.kernel_var]) if isinstance(self.kernel_var, (float, int)) else self.kernel_var
        lv = np.array([self.likeli_var]) if isinstance(self.likeli_var, (float, int)) else self.likeli_var

        # concat the parameters together
        try:
            x0 = np.concatenate([self.length_scales, kv, lv])
        except ValueError:
            # HACK: to deal with a dimension mis match
            x0 = np.concatenate([self.length_scales, np.array([kv]), np.array([lv])])

        # ------
        # transform the parameter values to variable values: optimise in variable space
        # ------

        x0 = self._apply_transform_funct(x0, func_type="inv_func")

        # run optimisation
        res = scipy.optimize.minimize(self.SMLII,
                                      x0=x0,
                                      args=(self.x, self.y[:, 0], False, None, jac),
                                      method=opt_method,
                                      jac=jac)

        # -----
        # transform variables back to parameters
        # -----

        pp_params = self._apply_transform_funct(res.x, func_type="func")

        # set hyper parameter values
        self.set_lengthscales(pp_params[:len(self.length_scales)])
        self.set_kernel_variance(pp_params[-2])
        self.set_likelihood_variance(pp_params[-1])

        # print(res["fun"])
        # return {"sucsess": res['success'], "marginal_loglikelihood_from_opt": res["fun"]}
        return res['success']

    def get_loglikelihood(self):
        kv = np.array([self.kernel_var]) if isinstance(self.kernel_var, (float, int)) else self.kernel_var
        lv = np.array([self.likeli_var]) if isinstance(self.likeli_var, (float, int)) else self.likeli_var

        kv = kv.reshape(1) if len(kv.shape) == 0 else kv
        lv = lv.reshape(1) if len(lv.shape) == 0 else lv

        hypers = np.concatenate([self.length_scales, kv, lv])

        # SMLII returns negative marginal log likelihood (when grad=False)
        # hyps = self._apply_transform_funct(hypers, func_type="func")
        # return - self.SMLII(hypers=hyps, x=self.x, y=self.y[:, 0], approx=False, M=None, grad=False)

        return - SMLII_mod(hypers=hypers, x=self.x, y=self.y[:, 0], approx=False, M=None, grad=False)

    def get_objective_function_value(self):
        # return the negative log likelihood
        return - self.get_loglikelihood()

    def SGPkernel(self, **kwargs):
        return SGPkernel(**kwargs)

    def SMLII(self, hypers, x, y, approx=False, M=None, grad=True):
        # this function takes in the variable representation of hyper parameters, which requires
        # a transform applied to get the parameter representation

        # convert the variable representation to (possibly) constrained parameter space
        hyps = self._apply_transform_funct(hypers, func_type="func")

        return SMLII_mod(hypers=hyps, x=x, y=y, approx=approx, M=M, grad=grad)




def SGPkernel(x, xs=None, grad=False, ell=1, sigma=1):
    """
    Return a Matern (3/2) covariance function for the given inputs.
    Inputs:
            x: training data of size n x 3 (3 corresponds to x,y,time)
            xs: test inputs of size ns x 3
            grad: Boolean whether to return the gradients of the covariance
                  function
            ell: correlation length-scales of the covariance function
            sigma: scaling pre-factor for covariance function
    Returns:
            sigma*k: scaled covariance function
            sigma*dk: scaled matrix of gradients
    """
    if xs is None:
        Q = squareform(pdist(np.sqrt(3.) * x / ell, 'euclidean'))
        k = (1 + Q) * np.exp(-Q)
        dk = np.zeros((len(ell), k.shape[0], k.shape[1]))
        for theta in range(len(ell)):
            q = squareform(pdist(np.sqrt(3.) * np.atleast_2d(x[:, theta] / ell[theta]).T, 'euclidean'))
            dk[theta, :, :] = q * q * np.exp(-Q)
    else:
        Q = cdist(np.sqrt(3.) * x / ell, np.sqrt(3.) * xs / ell, 'euclidean')
        k = (1 + Q) * np.exp(-Q)
    if grad:
        return sigma * k, sigma * dk
    else:
        return sigma * k


def Nystroem(x, y, M, ell, sf2, sn2, seed=20, opt=False):
    """
    Nyström approximation for kernel machines, e.g., Williams
    and Seeger, 2001. Produce a rank 'M' approximation of K
    and find its inverse via Woodbury identity. This is a
    faster approach of making predictions, but performance will
    depend on the value of M.
    """
    np.random.seed(seed)
    n = len(y)
    randselect = sorted(np.random.choice(range(n), M, replace=False))
    Kmm = SGPkernel(x[randselect, :], ell=ell, sigma=sf2)
    Knm = SGPkernel(x, xs=x[randselect, :], ell=ell, sigma=sf2)
    Vi = np.eye(n) / sn2

    s, u = np.linalg.eigh(Kmm)
    s[s <= 0] = 1e-12
    s_tilde = n * s / M
    u_tilde = np.sqrt(M / n) * np.dot(Knm, u) / s
    L = np.linalg.cholesky(np.diag(1 / s_tilde) + mdot([u_tilde.T, Vi, u_tilde]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.dot(u_tilde.T, Vi)))
    Ki = Vi - mdot([Vi, u_tilde, alpha])  # using Woodbury identity
    if opt:
        L_tilde = np.sqrt(s_tilde) * u_tilde
        det = np.linalg.slogdet(np.eye(M) * sn2 + np.dot(L_tilde.T, L_tilde))
        return Ki, np.atleast_2d(np.dot(Ki, y)).T, (det[0] * det[1]) / 2
    else:
        return Ki, np.atleast_2d(np.dot(Ki, y)).T



def SMLII_mod(hypers, x, y, approx=False, M=None, grad=True, use_log=True):
    """
    Objective function to minimise when optimising the model
    hyperparameters. This function is the negative log marginal likelihood.
    Inputs:
            hypers: initial guess of hyperparameters
            x: inputs (vector of size n x 3)
            y: outputs (freeboard values from all satellites, size n x 1)
            approx: Boolean, whether to use Nyström approximation method
            M: number of training points to use in Nyström approx (integer scalar)
    Returns:
            nlZ: negative log marginal likelihood
            dnLZ: gradients of the negative log marginal likelihood
    """
    # ell = [np.exp(hypers[0]), np.exp(hypers[1]), np.exp(hypers[2])]
    # sf2 = np.exp(hypers[3])
    # sn2 = np.exp(hypers[4])

    # ----
    # The hyper parameters are a transform of a variable
    # ----

    # apply the transform here
    # if use_log:
    #     ell = np.exp(hypers[:-2])
    #     sf2 = np.exp(hypers[-2])
    #     sn2 = np.exp(hypers[-1])
    # else:
    #     ell = hypers[:-2]
    #     sf2 = hypers[-2]
    #     sn2 = hypers[-1]

    # HARDCODED: number of length scales expected to be 2, follows by kernel then likelihood variance
    ell = hypers[:-2]
    sf2 = hypers[-2]
    sn2 = hypers[-1]


    n = len(y)
    Kx, dK = SGPkernel(x, grad=True, ell=ell, sigma=sf2)
    try:
        if approx:
            Ki, A, det = Nystroem(x, y, M=M, ell=ell, sf2=sf2, sn2=sn2, opt=True)
            nlZ = np.dot(y.T, A) / 2 + det + n * np.log(2 * np.pi) / 2
            Q = Ki - np.dot(A, A.T)
        else:
            L = np.linalg.cholesky(Kx + np.eye(n) * sn2)
            A = np.atleast_2d(np.linalg.solve(L.T, np.linalg.solve(L, y))).T
            nlZ = np.dot(y.T, A) / 2 + np.log(L.diagonal()).sum() + n * np.log(2 * np.pi) / 2
            Q = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n))) - np.dot(A, A.T)

        if grad:
            dnlZ = np.zeros(len(hypers))
            for theta in range(len(hypers)):
                if theta < (len(hypers) - 2):
                    dnlZ[theta] = (Q * dK[theta, :, :]).sum() / 2
                elif theta == (len(hypers) - 2):
                    dnlZ[theta] = (Q * (2 * Kx)).sum() / 2
                elif theta == (len(hypers) - 1):
                    dnlZ[theta] = sn2 * np.trace(Q)
    except np.linalg.LinAlgError as e:
        nlZ = np.inf;
        dnlZ = np.ones(len(hypers)) * np.inf

    if grad:
        return nlZ, dnlZ
    else:
        return nlZ


def GPR(x, y, xs, ell, sf2, sn2, mean, approx=False, M=None, returnprior=False):
    """
    Gaussian process regression function to predict radar freeboard
    Parameters
    ----------
        x: training data of size n x 3 (3 corresponds to x,y,time)
        y: training outputs of size n x 1 (observations of radar freeboard)
        xs: test inputs of size ns x 3
        ell: correlation length-scales of the covariance function (vector of length 3)
        sf2: scaling pre-factor for covariance function (scalar)
        sn2: noise variance (scalar)
        mean: prior mean (scalar)
        approx: Boolean, whether to use Nyström approximation method
        M: number of training points to use in Nyström approx (integer scalar)
    Returns
    -------
        fs: predictive mean
        sfs2: predictive variance
        np.sqrt(Kxs[0][0]): prior variance
    """
    n = len(y)
    Kxsx = SGPkernel(x, xs=xs, ell=ell, sigma=sf2)
    Kxs = SGPkernel(xs, ell=ell, sigma=sf2)

    if approx:
        if M is None:
            M = int(n / 5)
        Ki, A = Nystroem(x, y, M=M, ell=ell, sf2=sf2, sn2=sn2)
        err = mdot([Kxsx.T, Ki, Kxsx])
    else:
        # this algo follows Algo 2.1 in Rasmussen (2006)
        Kx = SGPkernel(x, ell=ell, sigma=sf2) + np.eye(n) * sn2
        L = np.linalg.cholesky(Kx)
        A = np.linalg.solve(L.T, np.linalg.solve(L, y))
        v = np.linalg.solve(L, Kxsx)
        err = np.dot(v.T, v)

    fs = mean + np.dot(Kxsx.T, A)
    # taking the square root makes it standard deviation
    # TODO: update doc string
    sfs2 = np.sqrt((Kxs - err).diagonal())
    if returnprior:
        return fs, sfs2, np.sqrt(Kxs.diagonal()) # np.sqrt(Kxs[0][0])
    else:
        return fs, sfs2


if __name__ == "__main__":

    from GPSat.plot_utils import plot_gpflow_minimal_example

    # run a simple example
    res = plot_gpflow_minimal_example(PurePythonGPR,
                                      model_init=None,
                                      opt_params=dict(opt_method="L-BFGS-B", jac=False),
                                      pred_params=None)

    # ----
    # compare to GPflow
    # ---

    # toy data
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

    # unconstrained (well, hyper parameters must be positive) example
    m = PurePythonGPR(coords=X,
                      obs=Y)
    m.optimise_parameters()
    m_obj = m.get_objective_function_value()
    print(m_obj)
    print(m.get_parameters())

    # compare with GPflow
    from GPSat.models import get_model
    gpf = get_model("GPflowGPRModel")

    m2 = gpf(coords=X, obs=Y)
    m2.optimise_parameters()
    print(m2.get_objective_function_value())
    print(m2.get_parameters())

    # compare the params
    opt_params2 = m2.get_parameters()
    print("difference in hyper parameters with GPflow")
    for k, v in m.get_parameters().items():
        print(f"{k}: {v- opt_params2[k]}")

    print("difference in objective function value:")
    print(m2.get_objective_function_value() - m.get_objective_function_value())

    # --
    # check is load parameters from PurePython
    # --

    m2_obj = m2.get_objective_function_value()

    m2.set_parameters(**m.get_parameters())

    print("difference in objective function value, after setting parameters from PurePython:")
    print(m2.get_objective_function_value() - m_obj)

    # ---
    # constraints / transforms
    # ---


    constraints_dict = {
        "lengthscales": {
            "func": "sigmoid",
            "low": 0.3,
            "high": 0.5
        },
        "kernel_variance": {
            "func": "sigmoid",
            "low": 0.2,
            "high": 0.8
        },
        "likelihood_variance": {
            "func": "softplus",
            "shift": 0.1
        }
    }

    # initialise the model
    m = PurePythonGPR(coords=X,
                      obs=Y,
                      constraints_dict=constraints_dict,
                      # start hyper parameters outside of constraint range
                      length_scales=0.25,
                      kernel_var=0.1,
                      likelihood_variance=0.05)

    # optimise
    m.optimise()
    opt_params = m.get_parameters()
    print(opt_params)
    print(m.get_objective_function_value())

    # initialise the model - but don't specify the constraints_dict, use default hyper parameters
    m = PurePythonGPR(coords=X,
                      obs=Y)

    # set constraints separately
    m.set_parameter_constraints(constraints_dict=constraints_dict)
    # optimise
    m.optimise()

    # compare the params
    opt_params2 = m.get_parameters()
    print("difference in hyper parameters")
    for k, v in opt_params.items():
        print(f"{k}: {v- opt_params2[k]}")


