import gpflow
from gpflow.models import GPR, GPModel, SVGP
from gpflow.models.util import data_input_to_tensor, InducingVariablesLike
from gpflow.kernels import LinearCoregionalization
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction, Zero
from check_shapes import inherit_check_shapes
import tensorflow as tf
from likelihoods import LinearModelLikelihood, NonlinearModelLikelihood, ForwardModelLikelihood
from utils import add_likelihood_noise_cov, multioutput_conditional
from typing import Union, Optional


class MultioutputGPR(GPR):
    # M: Training size, N: Test Size, L: Latent GP dimension, P: Observation dimension
    def __init__(self,
                 data,
                 kernel,
                 num_latent_gps,
                 mean_function = None,
                 noise_variance = None,
                 likelihood: LinearModelLikelihood = None):
        assert isinstance(likelihood, LinearModelLikelihood)
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

    @inherit_check_shapes
    def log_marginal_likelihood(self) -> tf.Tensor:
        X, Y = self.data
        H = self.likelihood.H
        K = self.kernel(X, full_cov=True, full_output_cov=True) # Shape (N, L, N, L)
        K = tf.transpose(K, [0,2,1,3]) # Shape (N, N, L, L)
        HKHt = H @ K @ tf.transpose(H) # Shape (N, N, P, P)
        ks = add_likelihood_noise_cov(HKHt, self.likelihood.variance) # Shape (N, N, P, P)
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

        f_mean_zero, f_var = multioutput_conditional(
            kmn, kmm, knn, err, H=self.likelihood.H, R=self.likelihood.variance, full_cov=full_cov
        )  # [N, P], [N, P] or [P, N, N]

        f_mean = f_mean_zero + self.mean_function(Xnew)

        if not full_output_cov:
            f_var = tf.linalg.diag_part(f_var)

        return f_mean, f_var


class MultioutputSVGP(SVGP):
    def __init__(
        self,
        kernel: LinearCoregionalization,
        likelihood: ForwardModelLikelihood,
        inducing_variable: InducingVariablesLike,
        *,
        mean_function: Optional[MeanFunction] = None,
        q_diag: bool = False,
        q_mu: Optional[tf.Tensor] = None,
        q_sqrt: Optional[tf.Tensor] = None,
        num_data: Optional[tf.Tensor] = None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
            GPflow objects
        - q_diag is a boolean. If True, the covariance is approximated by a
            diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
            the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
            (relevant when feeding in external minibatches)
        """
        assert isinstance(kernel, LinearCoregionalization)
        assert isinstance(likelihood, ForwardModelLikelihood)

        super().__init__(kernel,
                         likelihood,
                         inducing_variable,
                         mean_function=mean_function,
                         num_latent_gps=likelihood.latent_dim,
                         q_diag=q_diag,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         num_data=num_data)

    @inherit_check_shapes
    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_cov = self.predict_f(X, full_cov=False, full_output_cov=True)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_cov, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl


if __name__ == "__main__":
    # Test SVGP
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import multivariate_gaussian_log_density, ForwardModel
    from copy import copy

    xs = np.arange(0, 5, 0.1)
    ys = np.arange(0, 5, 0.1)

    Xs, Ys = np.meshgrid(xs, ys)

    xdim, ydim = Xs.shape

    X = np.stack([Xs.ravel(), Ys.ravel()], axis=1)

    # Generate toy data
    kern1 = gpflow.kernels.Matern52()
    kern2 = gpflow.kernels.Matern52()
    kern3 = gpflow.kernels.Matern52(lengthscales=10)

    K1xx = kern1(X, X)
    K2xx = kern2(X, X)
    K3xx = kern3(X, X)

    np.random.seed(1)
    W = np.random.randn(xdim, ydim, 3)

    Chol1 = np.linalg.cholesky(K1xx)
    g1 = Chol1 @ W[...,0].ravel()

    Chol2 = np.linalg.cholesky(K2xx)
    g2 = Chol2 @ W[...,1].ravel()

    Chol3 = np.linalg.cholesky(K3xx)
    g3 = Chol3 @ W[...,2].ravel()

    # Independent latent GPs
    g1 = g1.reshape(xdim, ydim)
    g2 = g2.reshape(xdim, ydim)
    g3 = g3.reshape(xdim, ydim)

    # Mixed latent GPs
    mix_mat = np.array([[0.7, 0.3, 0.2], [0.3, 0.7, 0.1], [0.2, 0.1, 0.7]])
    # mix_mat = np.array([[0.7, 0.3, 0.], [0.3, 0.7, 0.], [0., 0., 0.7]])

    g123 = np.stack([g1, g2, g3], axis=2)[..., None] # Shape (X, Y, 3, 1)
    f123 = mix_mat[None, None] @ g123 # Shape (X, Y, 3, 1)
    f123 = f123[...,0] # Shape (X, Y, 3)

    f1 = f123[..., 0]
    f2 = f123[..., 1]
    f3 = f123[..., 2]

    # Generate observations
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def forward(f):
        y1 = sigmoid(f[...,2]) * f[...,0] + 0.5 * f[...,1]
        y2 = f[...,1]
        return np.stack([y1, y2], axis=-1)

    c = 0.5
    y1 = sigmoid(f3) * f1 + 0.5 * f2
    y2 = f2[...]

    num_obs = 200
    obs_noise = 1e-2

    rng = np.random.default_rng(0)
    x_idxs = np.arange(xdim)
    y_idxs = np.arange(ydim)
    X_idxs, Y_idxs = np.meshgrid(x_idxs[1:-1], y_idxs[1:-1], indexing='ij')
    all_idxs = np.stack([X_idxs.flatten(), Y_idxs.flatten()], axis=1)
    idxs = rng.choice(all_idxs, num_obs, replace=False)

    x_train = [X.reshape(xdim, ydim, 2)[tuple(idx)] for idx in idxs]
    y_train = [(forward(f123[tuple(idx)])).squeeze() + np.sqrt(obs_noise)*np.random.randn(2) for idx in idxs]
    x_train = np.array(x_train)
    y_train = np.array(y_train)

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


    # Create list of kernels for each output
    kern_list = [
        gpflow.kernels.Matern52(),
        gpflow.kernels.Matern52(),
        gpflow.kernels.Matern52(lengthscales=10)
    ]
    # Create multi-output kernel from kernel list
    kernel = gpflow.kernels.LinearCoregionalization(
        kern_list, W=mix_mat
    )  # Notice that we initialise the mixing matrix W
    # initialisation of inducing input locations (M random points from the training inputs)
    M = 200
    Z = copy(x_train)
    np.random.shuffle(Z)
    Z = Z[:M]
    # create multi-output inducing variables from Z
    iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(Z)
    )

    forward_map = ExampleModel(c=0.5, input_dim=2, latent_dim=3, observation_dim=2)
    likelihood = NonlinearModelLikelihood(forward_map, 1e-2, num_samples=1000)

    # initialize mean of variational posterior to be of shape MxL
    q_mu = np.zeros((M, 3))
    # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(M)[None, ...], 3, axis=0) * 1.0

    # create SVGP model as usual and optimize
    m = MultioutputSVGP(
        kernel,
        likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )

    gpflow.utilities.set_trainable(m.kernel.W, False)
    gpflow.utilities.set_trainable(m.likelihood.variance, False) 

    def optimize_model_with_adam(model, data, iterations, minibatch_size=128):
        (X, Y) = data
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(X.shape[0])
        train_iter = iter(train_dataset.batch(minibatch_size))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam()
        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)
        for step in range(iterations):
            if step % 100 == 0:
                print(f"step: {step}, elbo: {-training_loss().numpy()}")
            optimization_step()

    optimize_model_with_adam(m, (x_train, y_train), 4000)

    f_mean, f_var = m.predict_f(X)

    f_mean = tf.reshape(f_mean, (xdim, ydim, 3))

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(f1)
    axs[0,0].set_title('f1 true')
    axs[0,1].imshow(f2)
    axs[0,1].set_title('f2 true')
    axs[0,2].imshow(f3)
    axs[0,2].set_title('f3 true')
    axs[1,0].imshow(f_mean[...,0])
    axs[1,0].set_title('f1 prediction')
    axs[1,1].imshow(f_mean[...,1])
    axs[1,1].set_title('f2 prediction')
    axs[1,2].imshow(f_mean[...,2])
    axs[1,2].set_title('f3 prediction')
    plt.tight_layout()

