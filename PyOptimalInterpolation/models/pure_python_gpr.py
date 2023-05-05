# module for a "Pure Python" implementation of GPR using a Matern 3/2 Kernel

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot as mdot
import scipy
from scipy.spatial.distance import squareform, pdist, cdist

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.models import BaseGPRModel



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
        # TODO: there should be a shape check here, lengthscales should match coords dim
        assert self.coords.shape[1] == len(lengthscales), "lengthscales must align to dim of coords"
        self.length_scales = lengthscales

    def set_kernel_variance(self, kernel_variance):
        self.kernel_var = kernel_variance

    def set_likelihood_variance(self, likelihood_variance):
        self.likeli_var = likelihood_variance

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
            "y": res[0].flatten(),
            "y_var": res[1] ** 2 + self.likeli_var
        }
        return out

    @timer
    def optimise_parameters(self, opt_method="L-BFGS-B", jac=False):
        """an inheriting class should define method for optimising (hyper/variational) parameters"""
        # NOTE: previous default params were: opt_method="CG", jac=True
        return self.optimise(opt_method=opt_method, jac=jac)

    def optimise(self, opt_method="L-BFGS-B", jac=False):
        kv = np.array([self.kernel_var]) if isinstance(self.kernel_var, (float, int)) else self.kernel_var
        lv = np.array([self.likeli_var]) if isinstance(self.likeli_var, (float, int)) else self.likeli_var

        try:
            x0 = np.concatenate([self.length_scales, kv, lv])
        except ValueError:
            # HACK: to deal with a dimension mis match
            x0 = np.concatenate([self.length_scales, np.array([kv]), np.array([lv])])

        # take the log of x0 because the first step in SMLII is to take exp
        x0 = np.log(x0)
        res = scipy.optimize.minimize(self.SMLII,
                                      x0=x0,
                                      args=(self.x, self.y[:, 0], False, None, jac),
                                      method=opt_method,
                                      jac=jac)

        # take exponential to 'de-log' parameters
        pp_params = np.exp(res.x)

        self.length_scales = pp_params[:len(self.length_scales)]
        self.kernel_var = pp_params[-2]
        self.likeli_var = pp_params[-1]

        # return {"sucsess": res['success'], "marginal_loglikelihood_from_opt": res["fun"]}
        return res['success']

    def get_loglikelihood(self):
        kv = np.array([self.kernel_var]) if isinstance(self.kernel_var, (float, int)) else self.kernel_var
        lv = np.array([self.likeli_var]) if isinstance(self.likeli_var, (float, int)) else self.likeli_var

        kv = kv.reshape(1) if len(kv.shape) == 0 else kv
        lv = lv.reshape(1) if len(lv.shape) == 0 else lv

        hypers = np.concatenate([self.length_scales, kv, lv])

        # SMLII returns negative marginal log likelihood (when grad=False)
        return -self.SMLII(hypers=np.log(hypers), x=self.x, y=self.y[:, 0], approx=False, M=None, grad=False)

    def get_objective_function_value(self):
        return self.get_loglikelihood()

    def SGPkernel(self, **kwargs):
        return SGPkernel(**kwargs)

    def SMLII(self, hypers, x, y, approx=False, M=None, grad=True):
        return SMLII_mod(hypers=hypers, x=x, y=y, approx=approx, M=M, grad=grad)




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

    if use_log:
        ell = np.exp(hypers[:-2])
        sf2 = np.exp(hypers[-2])
        sn2 = np.exp(hypers[-1])
    else:
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

    from PyOptimalInterpolation.plot_utils import plot_gpflow_minimal_example

    # run a simple example
    res = plot_gpflow_minimal_example(PurePythonGPR,
                                      model_init=None,
                                      opt_params=dict(opt_method="L-BFGS-B", jac=False),
                                      pred_params=None)
