import numpy as np
import tensorflow as tf
import gpflow
from gpflow.likelihoods import Likelihood, MonteCarloLikelihood
from gpflow.utilities.parameter_or_function import prepare_parameter_or_function
from check_shapes import check_shapes, inherit_check_shapes
from utils import multivariate_gaussian_log_density, ForwardModel, LinearForwardModel
from abc import ABC, abstractmethod
from typing import Union


# TODO: Rename variance -> obs_covariance / R?

class ForwardModelLikelihood(ABC):
    def __init__(self, variance: tf.Tensor, *args, **kwargs):
        self.h = self.get_model(*args, **kwargs)
        assert isinstance(self.h, ForwardModel)

        observation_dim = self.h.observation_dim
        if isinstance(variance, (int, float)):
            variance = variance*tf.eye(observation_dim, dtype=tf.float64)
            
        assert variance.shape == (observation_dim, observation_dim)

        self.variance = prepare_parameter_or_function(variance) # Converts variance to gpflow parameter

    @abstractmethod
    def get_model(self, *args, **kwargs) -> ForwardModel:
        return NotImplementedError


class LinearModelLikelihood(Likelihood, ForwardModelLikelihood):
    """
    Implements likelihood of form y = Hx + noise
    """
    def __init__(self,
                 input_dim: int,
                 variance: tf.Tensor,
                 forward_model: Union[tf.Tensor, LinearForwardModel]):
        
        ForwardModelLikelihood.__init__(self, variance, input_dim, forward_model)
        Likelihood.__init__(self,
                            input_dim=self.h.input_dim,
                            latent_dim=self.h.latent_dim,
                            observation_dim=self.h.observation_dim)

        self.H = self.h.tensor

    def get_model(self, input_dim, forward_model):
        if isinstance(forward_model, (tf.Tensor, np.ndarray)):
            observation_dim, latent_dim = forward_model.shape
            H = forward_model
            return LinearForwardModel(input_dim, latent_dim, observation_dim, H)
        elif isinstance(forward_model, LinearForwardModel):
            return forward_model
        else:
            return TypeError

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
        HF = self.h(X, F)
        # return gpflow.logdensities.gaussian(Y, HF, self.variance) # Note: only deals with scalar y
        return multivariate_gaussian_log_density(Y, HF, self.variance)

    @inherit_check_shapes
    def _conditional_mean(self, X, F) -> tf.Tensor:
        HF = self.h(X, F)
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
        HFmu = self.h.propagate_mean(Fmu)
        HFcovHt = self.h.propagate_cov(Fcov)
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
        HFmu = self.h.propagate_mean(Fmu) # Shape (..., P)
        HFcovHt = self.h.propagate_cov(Fcov) # Shape (..., P, P)
        return multivariate_gaussian_log_density(Y, HFmu, HFcovHt + self.variance)

    def predict_log_density(self, X, Fmu, Fcov, Y):
        return self._predict_log_density(X, Fmu, Fcov, Y)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fcov: [broadcast batch..., latent_dim, latent_dim]",
        "return: [batch...]",
    )
    def _variational_expectations(self, X, Fmu, Fcov, Y) -> tf.Tensor:
        HFmu = self.h.propagate_mean(Fmu) # Shape (..., P)
        HFcovHt = self.h.propagate_cov(Fcov) # Shape (..., P, P)
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


class NonlinearModelLikelihood(MonteCarloLikelihood, ForwardModelLikelihood):
    def __init__(self,
                 forward_model: ForwardModel,
                 variance: tf.Tensor,
                 num_samples: int=100) -> None:

        MonteCarloLikelihood.__init__(self,
                                      input_dim=forward_model.input_dim,
                                      latent_dim=forward_model.latent_dim,
                                      observation_dim=forward_model.observation_dim)

        ForwardModelLikelihood.__init__(self, variance, forward_model)

        self.num_monte_carlo_points = num_samples

    def get_model(self, forward_model):
        return forward_model

    @check_shapes(
        "Fmu: [batch..., latent_dim]",
        "Fcov: [batch..., latent_dim, latent_dim]",
        "Ys.values(): [batch..., .]",
        "return: [broadcast n_funcs, batch..., .]",
    ) # 
    def _mc_quadrature(self, func, Fmu, Fcov, **Ys) -> tf.Tensor:
        N, D = tf.shape(Fmu)[0], tf.shape(Fcov)[-1]
        S = self.num_monte_carlo_points
        epsilon = tf.random.normal(shape=[S, N, D], dtype=tf.float64)
        L = tf.linalg.cholesky(Fcov)
        mc_x = Fmu[None, :, :] + tf.reduce_mean(L[None, :, :, :] @ epsilon[..., None], -1)
        mc_Xr = tf.reshape(mc_x, (S * N, D))
        for name, Y in Ys.items():
            D_out = Y.shape[1]
            mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])  # [S, N, D_out]
            Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))  # [S * N, D_out]
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        return tf.reduce_mean(feval, axis=0)

    @inherit_check_shapes
    def _log_prob(self, X, F, Y) -> tf.Tensor:
        Y_pred = self.h(X, F)
        return multivariate_gaussian_log_density(Y, Y_pred, self.variance)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fcov: [broadcast batch..., latent_dim, latent_dim]",
        "return: [batch...]",
    )
    def _variational_expectations(self, X, Fmu, Fcov, Y) -> tf.Tensor:
        def log_prob(F, X_, Y_) -> tf.Tensor:
            return self.log_prob(X_, F, Y_)

        return tf.reduce_sum(
            self._mc_quadrature(log_prob, Fmu, Fcov, X_=X, Y_=Y), axis=-1
        )
    
    def variational_expectations(self, X, Fmu, Fcov, Y):
        return self._variational_expectations(X, Fmu, Fcov, Y)
   
        
if __name__ == "__main__":

    # Test
    lik = LinearModelLikelihood(input_dim=2, variance=0.1, forward_model=tf.constant([[0.5, 0.5]], dtype=tf.float64))

    f = tf.constant([[1., 1.]], dtype=tf.float64)
    fmu = tf.constant([[1., 1.]], dtype=tf.float64)
    fvar = tf.eye(2, dtype=tf.float64)[None]
    X = tf.constant([[0.5, 0.5]], dtype=tf.float64)
    y = tf.constant([[1.]], dtype=tf.float64)

    print(lik.conditional_mean(X, f))
    print(lik.conditional_variance(X, f))
    print(lik.predict_mean_and_var(X, fmu, fvar))
    print(lik.predict_log_density(X, fmu, fvar, y))
    print(lik.variational_expectations(X, fmu, fvar, y))

    class ExampleModel(ForwardModel):
        def __init__(self, c, *, input_dim, latent_dim, observation_dim):
            super().__init__(input_dim=input_dim, latent_dim=latent_dim, observation_dim=observation_dim)
            self.c = c

        @inherit_check_shapes
        def _forward(self, X, F):
            sigmoid = tf.math.sigmoid
            y1 = sigmoid(F[...,2]) * F[...,0] + self.c * F[...,1]
            y2 = F[...,1]
            return tf.stack([y1, y2], axis=-1)

    forward_map = ExampleModel(c=0.5, input_dim=2, latent_dim=3, observation_dim=2)
    lik = NonlinearModelLikelihood(forward_map, variance=0.1, num_samples=1000)

    f = tf.constant([[1., 1., 1.]], dtype=tf.float64)
    fmu = tf.constant([[1., 1., 1.]], dtype=tf.float64)
    fvar = tf.eye(3, dtype=tf.float64)[None]
    X = tf.constant([[0.5, 0.5]], dtype=tf.float64)
    y = tf.constant([[1., 1.]], dtype=tf.float64)

    print(lik.variational_expectations(X, fmu, fvar, y))


