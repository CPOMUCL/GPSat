import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import gpflow
from gpflow.likelihoods import Gaussian, Likelihood
from gpflow.models import GPR, GPModel
from gpflow.utilities.parameter_or_function import (
    ConstantOrFunction,
    ParameterOrFunction,
    evaluate_parameter_or_function,
    prepare_parameter_or_function,
)
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import add_likelihood_noise_cov
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction, Zero
from check_shapes import check_shapes, inherit_check_shapes
import tensorflow as tf
from typing import Optional
from copy import copy
from likelihoods import LinearModelLikelihood


class MultioutputGPR(GPR):
    # M: Training size, N: Test Size, L: Latent GP dimension, P: Observation dimension
    def __init__(self,
                 data,
                 kernel,
                 num_latent_gps,
                 mean_function = None,
                 noise_variance = None,
                 likelihood: LinearModelLikelihood = None):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data

        GPModel.__init__(self, kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = data_input_to_tensor(data)

        if mean_function is None:
            self.mean_function = Zero(output_dim=Y_data.shape[-1])

    @staticmethod
    @check_shapes(
        "K: [batch..., N, N, P, P]",
        "return: [batch..., N, N, P, P]",
    ) # TODO: Simplify by taking shape [N, P, N, P] instead
    def add_likelihood_noise_cov(K, likelihood: LinearModelLikelihood):
        R = likelihood.variance # Shape (P, P)
        N, P = K.shape[-3], K.shape[-1]
        # Construct a block-diagonal matrix of R's to add to K
        R_op = tf.linalg.LinearOperatorFullMatrix(R)
        R_op_diag = tf.linalg.LinearOperatorBlockDiag([R_op for _ in range(N)]) # Shape (N*P, N*P)
        # Reshape K to tensor of hape (..., N*P, N*P)
        axis_indices = [i for i in range(len(K.shape))]
        leading_indices, tensor_indices = axis_indices[:-4], axis_indices[-4:]
        tensor_indices_permuted = [tensor_indices[0], tensor_indices[2], tensor_indices[1], tensor_indices[3]]
        K = tf.transpose(K, leading_indices+tensor_indices_permuted) # Shape (..., N, P, N, P)
        leading_dims = K.shape[:-4]
        K = tf.reshape(K, leading_dims+[N*P, N*P]) # Shape (..., N*P, N*P)
        # Add R's to the diagonal of K and reshape
        KnR = R_op_diag.add_to_tensor(K) # Shape (..., N*P, N*P)
        KnR = tf.reshape(KnR, leading_dims+[N, P, N, P]) # Shape (..., N, P, N, P)
        return tf.transpose(KnR, leading_indices+tensor_indices_permuted) # Shape (..., N, N, P, P)

    @check_shapes(
        "Kmn: [M, L, N, L]",
        "Kmm: [M, L, M, L]",
        "Knn: [N, L, N, L] if full_cov",
        "Knn: [N, L, L] if not full_cov",
        "f: [M, P]",
        "return[0]: [N, L]",
        "return[1]: [N, L, N, L] if full_cov",
        "return[1]: [N, L, L] if not full_cov",
    )
    def multioutput_conditional(self,
                                Kmn: tf.Tensor, # Shape (M, L, N, L)
                                Kmm: tf.Tensor, # Shape (M, L, M, L)
                                Knn: tf.Tensor, # Shape (N, L, N, L)
                                f: tf.Tensor, # Shape (M, P)
                                *,
                                full_cov=False):

        Kmn = tf.transpose(Kmn, [0,2,1,3]) # Shape (M, N, L, L)
        Kmm = tf.transpose(Kmm, [0,2,1,3]) # Shape (M, M, L, L)
        H = self.likelihood.H

        HKmmHt = H @ Kmm @ tf.transpose(H) # Shape (M, M, P, P)
        HKmn = H @ Kmn # Shape (M, N, P, L)
        ks = self.add_likelihood_noise_cov(HKmmHt, self.likelihood) # Shape (M, M, P, P)
        ks = tf.transpose(ks, [0,2,1,3]) # Shape (M, P, M, P)
        N, P, M, _ = ks.shape
        ks = tf.reshape(ks, (M*P, M*P)) # Reshape to size (MP, MP)
        Lm = tf.linalg.cholesky(ks) # Shape (MP, MP)

        # Compute the projection matrix A
        HKmn = tf.transpose(HKmn, [0,2,1,3]) # Shape (M, P, N, L)
        M, P, N, L = HKmn.shape
        HKmn = tf.reshape(HKmn, (M*P, N*L)) # Reshape to size (MP, NL)
        A = tf.linalg.triangular_solve(Lm, HKmn, lower=True)  # Shape (MP, NL)

        # Compute conditional covariance
        if full_cov:
            Knn = tf.reshape(Knn, (N*L, N*L))
            fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True) # Shape (NL, NL)
            fvar = tf.reshape(fvar, (N, L, N, L))
        else:
            def square(tensor):
                """Input is a tensor of shape (M, P, L) """
                tensor = tf.reshape(tensor, (M*P, L)) # Reshape to size (MP, L)
                return tf.linalg.matmul(tensor, tensor, transpose_a=True) # Shape (L, L)

            A = tf.reshape(A, (M, P, N, L))
            AtA_diag = tf.stack([square(A[:,:,n,:]) for n in range(N)], 0) # Shape (N, L, L)
            fvar = Knn - AtA_diag
            A = tf.reshape(A, (M*P, N*L))

        # Compute conditional mean
        f = tf.reshape(f, (M*P,1)) # Shape (MP, 1)
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False) # Shape (MP, NL)
        fmean = tf.linalg.matmul(A, f, transpose_a=True) # Shape (NL, 1)
        fmean = tf.reshape(fmean, (N,L)) # Shape (N, L)

        return fmean, fvar

    @inherit_check_shapes
    def log_marginal_likelihood(self) -> tf.Tensor:
        X, Y = self.data
        H = self.likelihood.H
        K = self.kernel(X, full_cov=True, full_output_cov=True) # Shape (N, L, N, L)
        K = tf.transpose(K, [0,2,1,3]) # Shape (N, N, L, L)
        HKHt = H @ K @ tf.transpose(H) # Shape (N, N, P, P)
        ks = self.add_likelihood_noise_cov(HKHt, self.likelihood) # Shape (N, N, P, P)
        ks = tf.transpose(ks, [0,2,1,3]) # Shape (N, P, N, P)
        N, P = ks.shape[:2]
        ks = tf.reshape(ks, (N*P, N*P)) # Shape (NP, NP)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)
        log_prob = multivariate_normal(tf.reshape(Y, (N*P, 1)),
                                       tf.reshape(m, (N*P, 1)),
                                       L
                    )
        return tf.reduce_sum(log_prob)

    @inherit_check_shapes
    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ):
        X, Y = self.data
        err = Y - self.mean_function(X) # TODO: Handle multivariate mean properly. Shape (M, P)

        kmm = self.kernel(X, full_cov=True, full_output_cov=True) # Shape (M, L, M, L)
        knn = self.kernel(Xnew, full_cov=full_cov, full_output_cov=True) # Shape (N, L, N, L)
        kmn = self.kernel(X, Xnew, full_cov=True, full_output_cov=True) # Shape (M, L, N, L)

        f_mean_zero, f_var = self.multioutput_conditional(
            kmn, kmm, knn, err, full_cov=full_cov
        )  # [N, P], [N, P] or [P, N, N]

        f_mean = f_mean_zero + self.mean_function(Xnew)

        if not full_output_cov:
            f_var = tf.linalg.diag_part(f_var)

        return f_mean, f_var


class MultioutputSVGP(gpflow.models.SVGP):
    @inherit_check_shapes
    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        if isinstance(self.likelihood, LinearModelLikelihood):
            full_cov = False
            full_output_cov = True
        else:
            full_cov = False
            full_output_cov=False
        f_mean, f_cov = self.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_cov, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

