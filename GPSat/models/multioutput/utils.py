import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from check_shapes import check_shapes, inherit_check_shapes
from typing import Union

# ----- Forward model class defining the likelihood -----

class ForwardModel(ABC):
    def __init__(self, input_dim: int, latent_dim: int, observation_dim: int, *args, **kwargs):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
    
    @abstractmethod
    @check_shapes(
        "X: [batch..., input_dim]",
        "F: [batch..., latent_dim]",
        "return: [batch..., observation_dim]",
    )
    def _forward(self, X: tf.Tensor, F: tf.Tensor):
        """
        To implement the forward operation h(x) in the model-based likelihood
        """
        return NotImplementedError

    def __call__(self, X: tf.Tensor, F: tf.Tensor):
        return self._forward(X, F)


class LinearForwardModel(ForwardModel):
    """
    Class for linear forward models
    y = Hx
    where H is a P x L matrix
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 observation_dim: int,
                 H: Union[tf.Tensor, np.ndarray]):

        super().__init__(input_dim, latent_dim, observation_dim)
        assert H.shape == (observation_dim, latent_dim), "Tensor H in forward model must be of shape (obs_dim, latent_dim)"
        self.tensor = tf.constant(H)

    @inherit_check_shapes
    def _forward(self, X: tf.Tensor, F: tf.Tensor):
        return tf.linalg.matvec(self.tensor, F)

    def propagate_mean(self, Fmu: tf.Tensor):
        """
        Computes the first moment ∫ Hx p(x)dx = Hμ
        where μ is the mean of x
        """
        return self._forward(None, Fmu)

    def propagate_cov(self, Fcov: tf.Tensor):
        """
        Computes the second moment ∫ (H(x-μ))'(H(x-μ)) p(x)dx = HΣH'
        where Σ is the covariance of x
        """
        return self.tensor @ Fcov @ tf.transpose(self.tensor)


# ----- Functions for multivariate Gaussian manipulations -----

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
    "K: [batch..., N, N, P, P]",
    "R: [P, P]",
    "return: [batch..., N, N, P, P]",
) # TODO: Simplify by taking shape [N, P, N, P] instead
def add_likelihood_noise_cov(K, R):
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
    "H: [P, L]",
    "R: [P, P]",
    "return[0]: [N, L]",
    "return[1]: [N, L, N, L] if full_cov",
    "return[1]: [N, L, L] if not full_cov",
)
def multioutput_conditional(Kmn: tf.Tensor,
                            Kmm: tf.Tensor,
                            Knn: tf.Tensor,
                            f: tf.Tensor,
                            H: tf.Tensor,
                            R: tf.Tensor,
                            *,
                            full_cov=False):

    Kmn = tf.transpose(Kmn, [0,2,1,3]) # Shape (M, N, L, L)
    Kmm = tf.transpose(Kmm, [0,2,1,3]) # Shape (M, M, L, L)

    HKmmHt = H @ Kmm @ tf.transpose(H) # Shape (M, M, P, P)
    HKmn = H @ Kmn # Shape (M, N, P, L)
    ks = add_likelihood_noise_cov(HKmmHt, R) # Shape (M, M, P, P)
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

