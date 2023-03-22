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
from gpflow.likelihoods import MonteCarloLikelihood
from dataclasses import dataclass
from abc import ABC, abstractmethod


class LinearModelLikelihood(Likelihood):
    """
    Implements likelihood of form y = Hx + noise
    """
    def __init__(self, input_dim, variance, H):
        observation_dim, latent_dim = H.shape
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, observation_dim=observation_dim)
        
        if isinstance(variance, (int, float)):
            variance = variance*tf.eye(observation_dim, dtype=tf.float64)
            
        assert variance.shape == (observation_dim, observation_dim)

        self.variance = prepare_parameter_or_function(variance) # Converts variance to gpflow parameter
        self.H = H

    @staticmethod
    @check_shapes(
        "x: [broadcast shape..., input_dim]",
        "mu: [broadcast shape..., P]",
        "cov: [broadcast shape..., P, P]",
        "return: [shape...]",
    )
    def multivariate_gaussian_log_density(x, mu, cov) -> tf.Tensor:
        P = mu.shape[-1]
        diff = tf.expand_dims(mu-x, axis=-1) # (..., P, 1)
        shape = tf.concat([tf.shape(mu)[:-1], [P, P]], axis=0)
        cov_broadcast = tf.broadcast_to(cov, shape) # (..., P, P)
        out = - (P/2) * np.log(2 * np.pi) \
              - 0.5 * tf.linalg.logdet(cov) \
              - 0.5 * tf.linalg.matmul(diff, tf.linalg.solve(cov_broadcast, diff), transpose_a=True) # (..., 1, 1)
        return tf.reduce_sum(out, [-1,-2]) # (..., 1, 1) -> (...)

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N, P, P]",
    )
    def variance_at(self, X) -> tf.Tensor:
        variance = self.variance
        P = variance.shape[0]
        shape = tf.concat([tf.shape(X)[:-1], [P, P]], 0)
        return tf.broadcast_to(variance, shape)

    @inherit_check_shapes
    def _log_prob(self, X, F, Y) -> tf.Tensor:
        HF = tf.linalg.matvec(self.H, F)
        # return gpflow.logdensities.gaussian(Y, HF, self.variance) # Note: only deals with scalar y
        return self.multivariate_gaussian_log_density(Y, HF, self.variance)

    @inherit_check_shapes
    def _conditional_mean(self, X, F) -> tf.Tensor:
        HF = tf.linalg.matvec(self.H, F)
        return tf.identity(HF)

    @inherit_check_shapes
    def _conditional_variance(self, X, F) -> tf.Tensor:
        shape = (tf.shape(F)[0], self.observation_dim)
        return tf.broadcast_to(self.variance, shape)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fcov: [broadcast batch..., latent_dim, latent_dim]",
        "return[0]: [batch..., observation_dim]",
        "return[1]: [batch..., observation_dim, observation_dim]",
    )
    def _predict_mean_and_var(self, X, Fmu, Fcov) -> tf.Tensor:
        HFmu = tf.linalg.matvec(self.H, Fmu)
        HFcovHt = self.H @ Fcov @ tf.transpose(self.H)
        return tf.identity(HFmu), HFcovHt + self.variance

    def predict_mean_and_var(self, X, Fmu, Fcov):
        return self._predict_mean_and_var(X, Fmu, Fcov)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fcov: [broadcast batch..., latent_dim, latent_dim]",
        "return: [batch...]",
    )
    def _predict_log_density(self, X, Fmu, Fcov, Y) -> tf.Tensor:
        HFmu = tf.linalg.matvec(self.H, Fmu) # Shape (..., P)
        HFcovHt = self.H @ Fcov @ tf.transpose(self.H) # Shape (..., P, P)
        return self.multivariate_gaussian_log_density(Y, HFmu, HFcovHt + self.variance)

    def predict_log_density(self, X, Fmu, Fcov, Y):
        return self._predict_log_density(X, Fmu, Fcov, Y)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fcov: [broadcast batch..., latent_dim, latent_dim]",
        "return: [batch...]",
    )
    def _variational_expectations(self, X, Fmu, Fcov, Y) -> tf.Tensor:
        HFmu = tf.linalg.matvec(self.H, Fmu) # Shape (..., P)
        HFcovHt = self.H @ Fcov @ tf.transpose(self.H) # Shape (..., P, P)
        covariance = self.variance # (P, P)
        shape = tf.concat([tf.shape(HFmu)[:-1], tf.shape(covariance)], axis=0)
        cov_broadcast = tf.broadcast_to(covariance, shape) # (..., P, P)
        P = self.observation_dim
        diff = tf.expand_dims(Y-HFmu, axis=-1) # Shape (..., P, 1)
        output = - (P/2) * np.log(2 * np.pi) \
                 - 0.5 * tf.linalg.logdet(covariance) \
                 - 0.5 * tf.squeeze(tf.matmul(diff, tf.linalg.solve(cov_broadcast, diff), transpose_a=True)) \
                 - 0.5 * tf.linalg.trace(tf.linalg.solve(cov_broadcast, HFcovHt))
        return output

    def variational_expectations(self, X, Fmu, Fcov, Y):
        return self._variational_expectations(X, Fmu, Fcov, Y)


class ForwardModel(ABC):
    def __init__(self, input_dim, latent_dim, observation_dim, *args, **kwargs):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
    
    @abstractmethod
    @check_shapes(
        "X: [batch..., input_dim]",
        "F: [batch..., latent_dim]",
        "return: [batch..., observation_dim]",
    )
    def _forward(self, X, F):
        pass

    def __call__(self, X, F):
        return self._forward(X, F)


class NonlinearModelLikelihood(MonteCarloLikelihood):
    def __init__(self,
                 forward_map: ForwardModel,
                 variance: tf.Tensor,
                 num_samples: int=100) -> None:

        super().__init__(input_dim=forward_map.input_dim,
                         latent_dim=forward_map.latent_dim,
                         observation_dim=forward_map.observation_dim)

        self.h = forward_map
        if isinstance(variance, (int, float)):
            variance = variance*tf.eye(self.observation_dim, dtype=tf.float64)
            
        assert variance.shape == (self.observation_dim, self.observation_dim)

        self.variance = prepare_parameter_or_function(variance) # Converts variance to gpflow parameter
        self.num_monte_carlo_points = num_samples

    @inherit_check_shapes
    def _log_prob(self, X, F, Y) -> tf.Tensor:
        Y_pred = self.h(X, F)
        return LinearModelLikelihood.multivariate_gaussian_log_density(Y, Y_pred, self.variance)

