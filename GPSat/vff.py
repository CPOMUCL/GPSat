"""
Code adapted from:
https://github.com/st--/VFF
"""
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import default_float
from functools import reduce


# ----- matrix_structures -----

class DiagMat:
    def __init__(self, d):
        self.d = d

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    @property
    def sqrt_dims(self):
        return tf.size(self.d)

    def get(self):
        return tf.linalg.diag(self.d)

    def logdet(self):
        return tf.reduce_sum(tf.math.log(self.d))

    def matmul(self, B):
        return tf.expand_dims(self.d, 1) * B

    def solve(self, B):
        return B / tf.expand_dims(self.d, 1)

    def inv(self):
        return DiagMat(tf.math.reciprocal(self.d))

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        return tf.reduce_sum(tf.linalg.diag_part(X) / self.d)

    def get_diag(self):
        return self.d

    def inv_diag(self):
        return 1.0 / self.d

    def matmul_sqrt(self, B):
        return tf.expand_dims(tf.sqrt(self.d), 1) * B

    def matmul_sqrt_transpose(self, B):
        return tf.expand_dims(tf.sqrt(self.d), 1) * B


class LowRankMatNeg:
    def __init__(self, d, W):
        """
        A matrix of the form

            diag(d) - W W^T

        (note the minus sign)
        """
        self.d = d
        self.W = W

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    def get(self):
        return tf.linalg.diag(self.d) - tf.matmul(self.W, self.W, transpose_b=True)


class Rank1MatNeg:
    def __init__(self, d, v):
        """
        A matrix of the form

            diag(d) - v v^T

        (note the minus sign)
        """
        self.d = d
        self.v = v

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    def get(self):
        W = tf.expand_dims(self.v, 1)
        return tf.linalg.diag(self.d) - tf.matmul(W, W, transpose_b=True)


class Rank1Mat:
    def __init__(self, d, v):
        """
        A matrix of the form

            diag(d) + v v^T

        """
        self.d = d
        self.v = v

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    @property
    def sqrt_dims(self):
        return tf.size(self.d) + 1

    def get(self):
        V = tf.expand_dims(self.v, 1)
        return tf.linalg.diag(self.d) + tf.matmul(V, V, transpose_b=True)

    def logdet(self):
        return tf.reduce_sum(tf.math.log(self.d)) + tf.math.log(
            1.0 + tf.reduce_sum(tf.square(self.v) / self.d)
        )

    def matmul(self, B):
        V = tf.expand_dims(self.v, 1)
        return tf.expand_dims(self.d, 1) * B + tf.matmul(V, tf.matmul(V, B, transpose_a=True))

    def solve(self, B):
        div = self.v / self.d
        c = 1.0 + tf.reduce_sum(div * self.v)
        div = tf.expand_dims(div, 1)
        return B / tf.expand_dims(self.d, 1) - tf.matmul(
            div / c, tf.matmul(div, B, transpose_a=True)
        )

    def inv(self):
        di = tf.math.reciprocal(self.d)
        Div = self.v * di
        M = 1.0 + tf.reduce_sum(Div * self.v)
        v_new = Div / tf.sqrt(M)
        return Rank1MatNeg(di, v_new)

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        R = tf.expand_dims(self.v / self.d, 1)
        RTX = tf.matmul(R, X, transpose_a=True)
        RTXR = tf.matmul(RTX, R)
        M = 1 + tf.reduce_sum(tf.square(self.v) / self.d)
        return tf.reduce_sum(tf.linalg.diag_part(X) / self.d) - RTXR / M

    def get_diag(self):
        return self.d + tf.square(self.v)

    def inv_diag(self):
        div = self.v / self.d
        c = 1.0 + tf.reduce_sum(div * self.v)
        return 1.0 / self.d - tf.square(div) / c

    def matmul_sqrt(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   V^T  ]

        This method right-multiplies the sqrt by the matrix B
        """

        DB = tf.expand_dims(tf.sqrt(self.d), 1) * B
        VTB = tf.matmul(tf.expand_dims(self.v, 0), B)
        return tf.concat([DB, VTB], axis=0)

    def matmul_sqrt_transpose(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   W^T  ]

        This method right-multiplies the transposed-sqrt by the matrix B
        """
        B1 = tf.slice(B, tf.zeros((2,), tf.int32), tf.stack([tf.size(self.d), -1]))
        B2 = tf.slice(B, tf.stack([tf.size(self.d), 0]), -tf.ones((2,), tf.int32))
        return tf.expand_dims(tf.sqrt(self.d), 1) * B1 + tf.matmul(tf.expand_dims(self.v, 1), B2)


class LowRankMat:
    def __init__(self, d, W):
        """
        A matrix of the form

            diag(d) + W W^T

        """
        self.d = d
        self.W = W

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    @property
    def sqrt_dims(self):
        return tf.size(self.d) + tf.shape(self.W)[1]

    def get(self):
        return tf.linalg.diag(self.d) + tf.matmul(self.W, self.W, transpose_b=True)

    def logdet(self):
        part1 = tf.reduce_sum(tf.math.log(self.d))
        I = tf.eye(tf.shape(self.W)[1], dtype=default_float())
        M = I + tf.matmul(tf.transpose(self.W) / self.d, self.W)  # XXX
        part2 = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(M))))
        return part1 + part2

    def matmul(self, B):
        WTB = tf.matmul(self.W, B, transpose_a=True)
        WWTB = tf.matmul(self.W, WTB)
        DB = tf.reshape(self.d, [-1, 1]) * B
        return DB + WWTB

    def get_diag(self):
        return self.d + tf.reduce_sum(tf.square(self.W), 1)

    def solve(self, B):
        d_col = tf.expand_dims(self.d, 1)
        DiB = B / d_col
        DiW = self.W / d_col
        WTDiB = tf.matmul(DiW, B, transpose_a=True)
        I = tf.eye(tf.shape(self.W)[1], dtype=default_float())
        M = I + tf.matmul(DiW, self.W, transpose_a=True)
        L = tf.linalg.cholesky(M)
        Minv_WTDiB = tf.linalg.cholesky_solve(L, WTDiB)
        return DiB - tf.matmul(DiW, Minv_WTDiB)

    def inv(self):
        di = tf.math.reciprocal(self.d)
        d_col = tf.expand_dims(self.d, 1)
        DiW = self.W / d_col
        I = tf.eye(tf.shape(self.W)[1], dtype=default_float())
        M = I + tf.matmul(DiW, self.W, transpose_a=True)
        L = tf.linalg.cholesky(M)
        v = tf.transpose(tf.linalg.triangular_solve(L, tf.transpose(DiW), lower=True))  # XXX
        return LowRankMatNeg(di, v)

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        d_col = tf.expand_dims(self.d, 1)
        R = self.W / d_col
        RTX = tf.matmul(R, X, transpose_a=True)
        RTXR = tf.matmul(RTX, R)
        I = tf.eye(tf.shape(self.W)[1], dtype=default_float())
        M = I + tf.matmul(R, self.W, transpose_a=True)
        Mi = tf.linalg.inv(M)
        return tf.reduce_sum(tf.linalg.diag_part(X) * 1.0 / self.d) - tf.reduce_sum(RTXR * Mi)

    def inv_diag(self):
        d_col = tf.expand_dims(self.d, 1)
        WTDi = tf.transpose(self.W / d_col)  # XXX
        I = tf.eye(tf.shape(self.W)[1], dtype=default_float())
        M = I + tf.matmul(WTDi, self.W)
        L = tf.linalg.cholesky(M)
        tmp1 = tf.linalg.triangular_solve(L, WTDi, lower=True)
        return 1.0 / self.d - tf.reduce_sum(tf.square(tmp1), 0)

    def matmul_sqrt(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   W^T  ]

        This method right-multiplies the sqrt by the matrix B
        """

        DB = tf.expand_dims(tf.sqrt(self.d), 1) * B
        VTB = tf.matmul(self.W, B, transpose_a=True)
        return tf.concat([DB, VTB], axis=0)

    def matmul_sqrt_transpose(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   W^T  ]

        This method right-multiplies the transposed-sqrt by the matrix B
        """
        B1 = tf.slice(B, tf.zeros((2,), tf.int32), tf.stack([tf.size(self.d), -1]))
        B2 = tf.slice(B, tf.stack([tf.size(self.d), 0]), -tf.ones((2,), tf.int32))
        return tf.expand_dims(tf.sqrt(self.d), 1) * B1 + tf.matmul(self.W, B2)


class BlockDiagMat:
    def __init__(self, A, B):
        self.A, self.B = A, B

    @property
    def shape(self):
        mats = [self.A, self.B]
        return (sum([m.shape[0] for m in mats]), sum([m.shape[1] for m in mats]))

    @property
    def sqrt_dims(self):
        mats = [self.A, self.B]
        return sum([m.sqrt_dims for m in mats])

    def _get_rhs_slices(self, X):
        # X1 = X[:self.A.shape[1], :]
        X1 = tf.slice(X, begin=tf.zeros((2,), tf.int32), size=tf.stack([self.A.shape[1], -1]))
        # X2 = X[self.A.shape[1]:, :]
        X2 = tf.slice(X, begin=tf.stack([self.A.shape[1], 0]), size=-tf.ones((2,), tf.int32))
        return X1, X2

    def get(self):
        tl_shape = tf.stack([self.A.shape[0], self.B.shape[1]])
        br_shape = tf.stack([self.B.shape[0], self.A.shape[1]])
        top = tf.concat([self.A.get(), tf.zeros(tl_shape, default_float())], axis=1)
        bottom = tf.concat([tf.zeros(br_shape, default_float()), self.B.get()], axis=1)
        return tf.concat([top, bottom], axis=0)

    def logdet(self):
        return self.A.logdet() + self.B.logdet()

    def matmul(self, X):
        X1, X2 = self._get_rhs_slices(X)
        top = self.A.matmul(X1)
        bottom = self.B.matmul(X2)
        return tf.concat([top, bottom], axis=0)

    def solve(self, X):
        X1, X2 = self._get_rhs_slices(X)
        top = self.A.solve(X1)
        bottom = self.B.solve(X2)
        return tf.concat([top, bottom], axis=0)

    def inv(self):
        return BlockDiagMat(self.A.inv(), self.B.inv())

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        X1, X2 = tf.slice(X, [0, 0], self.A.shape), tf.slice(X, self.A.shape, [-1, -1])
        top = self.A.trace_KiX(X1)
        bottom = self.B.trace_KiX(X2)
        return top + bottom

    def get_diag(self):
        return tf.concat([self.A.get_diag(), self.B.get_diag()], axis=0)

    def inv_diag(self):
        return tf.concat([self.A.inv_diag(), self.B.inv_diag()], axis=0)

    def matmul_sqrt(self, X):
        X1, X2 = self._get_rhs_slices(X)
        top = self.A.matmul_sqrt(X1)
        bottom = self.B.matmul_sqrt(X2)
        return tf.concat([top, bottom], axis=0)

    def matmul_sqrt_transpose(self, X):
        X1 = tf.slice(X, begin=tf.zeros((2,), tf.int32), size=tf.stack([self.A.sqrt_dims, -1]))
        X2 = tf.slice(X, begin=tf.stack([self.A.sqrt_dims, 0]), size=-tf.ones((2,), tf.int32))
        top = self.A.matmul_sqrt_transpose(X1)
        bottom = self.B.matmul_sqrt_transpose(X2)

        return tf.concat([top, bottom], axis=0)


# ---- spectral_covariance ----

def make_Kuu(kern, a, b, ms):
    """
    # Make a representation of the Kuu matrices
    """
    omegas = 2.0 * np.pi * ms / (b - a)
    #omegas = omegas.astype(np.float64)
    if isinstance(kern, gpflow.kernels.Matern12):
        # cos part first
        lamb = 1.0 / kern.lengthscales
        two_or_four = np.where(omegas == 0, 2.0, 4.0)
        d_cos = (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kern.variance / two_or_four
        v_cos = tf.ones(tf.shape(d_cos), default_float()) / tf.sqrt(kern.variance)

        # now the sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        d_sin = (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kern.variance / 4.0

        return BlockDiagMat(Rank1Mat(d_cos, v_cos), DiagMat(d_sin))

    elif isinstance(kern, gpflow.kernels.Matern32):
        # cos part first
        lamb = np.sqrt(3.0) / kern.lengthscales
        four_or_eight = np.where(omegas == 0, 4.0, 8.0)
        d_cos = (
            (b - a)
            * tf.square(tf.square(lamb) + tf.cast(tf.square(omegas), dtype=tf.float64))
            / tf.pow(lamb, 3)
            / kern.variance
            / four_or_eight
        )
        v_cos = tf.ones(tf.shape(d_cos), tf.float64) / tf.sqrt(kern.variance)

        # now the sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        d_sin = (
            (b - a)
            * tf.square(tf.square(lamb) + tf.cast(tf.square(omegas), dtype=tf.float64))
            / tf.pow(lamb, 3)
            / kern.variance
            / 8.0
        )
        v_sin = omegas / lamb / tf.sqrt(kern.variance)
        return BlockDiagMat(Rank1Mat(d_cos, v_cos), Rank1Mat(d_sin, v_sin))

    elif isinstance(kern, gpflow.kernels.Matern52):
        # cos part:
        lamb = np.sqrt(5.0) / kern.lengthscales
        sixteen_or_32 = np.where(omegas == 0, 16.0, 32.0)
        v1 = (3 * tf.square(omegas / lamb) - 1) / tf.sqrt(8 * kern.variance)
        v2 = tf.ones(tf.shape(v1), default_float()) / tf.sqrt(kern.variance)
        W_cos = tf.concat([tf.expand_dims(v1, 1), tf.expand_dims(v2, 1)], axis=1)
        d_cos = (
            3
            * (b - a)
            / sixteen_or_32
            / tf.pow(lamb, 5)
            / kern.variance
            * tf.pow(tf.square(lamb) + tf.square(omegas), 3)
        )

        # sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        v_sin = np.sqrt(3.0) * omegas / lamb / tf.sqrt(kern.variance)
        d_sin = (
            3
            * (b - a)
            / 32.0
            / tf.pow(lamb, 5)
            / kern.variance
            * tf.pow(tf.square(lamb) + tf.square(omegas), 3)
        )
        return BlockDiagMat(LowRankMat(d_cos, W_cos), Rank1Mat(d_sin, v_sin))
    else:
        raise NotImplementedError


def make_Kuf(k, X, a, b, ms):
    omegas = 2.0 * np.pi * ms / (b - a)
    # if default_float() is np.float32:
    #     omegas = omegas.astype(np.float32)
    Kuf_cos = tf.transpose(tf.cos(omegas * (X - a)))
    omegas_sin = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas_sin * (X - a)))

    # correct Kfu outside [a, b]
    lt_a_sin = tf.tile(tf.transpose(X) < a, [len(ms) - 1, 1])
    gt_b_sin = tf.tile(tf.transpose(X) > b, [len(ms) - 1, 1])
    lt_a_cos = tf.tile(tf.transpose(X) < a, [len(ms), 1])
    gt_b_cos = tf.tile(tf.transpose(X) > b, [len(ms), 1])
    if isinstance(k, gpflow.kernels.Matern12):
        # Kuf_sin[:, np.logical_or(X.flatten() < a, X.flatten() > b)] = 0
        Kuf_sin = tf.where(
            tf.logical_or(lt_a_sin, gt_b_sin), tf.zeros(tf.shape(Kuf_sin), default_float()), Kuf_sin
        )
        Kuf_cos = tf.where(
            lt_a_cos,
            tf.tile(tf.exp(-tf.abs(tf.transpose(X - a)) / k.lengthscales), [len(ms), 1]),
            Kuf_cos,
        )
        Kuf_cos = tf.where(
            gt_b_cos,
            tf.tile(tf.exp(-tf.abs(tf.transpose(X - b)) / k.lengthscales), [len(ms), 1]),
            Kuf_cos,
        )
    elif isinstance(k, gpflow.kernels.Matern32):
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = tf.tile((1 + arg) * tf.exp(-arg), [len(ms), 1])
        Kuf_cos = tf.where(lt_a_cos, edge, Kuf_cos)
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - b) / k.lengthscales
        edge = tf.tile((1 + arg) * tf.exp(-arg), [len(ms), 1])
        Kuf_cos = tf.where(gt_b_cos, edge, Kuf_cos)

        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = (tf.transpose(X) - a) * tf.exp(-arg) * omegas_sin[:, None]
        Kuf_sin = tf.where(lt_a_sin, edge, Kuf_sin)
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - b) / k.lengthscales
        edge = (tf.transpose(X) - b) * tf.exp(-arg) * omegas_sin[:, None]
        Kuf_sin = tf.where(gt_b_sin, edge, Kuf_sin)
    elif isinstance(k, gpflow.kernels.Matern52):
        # edges not implemented yet
        tf.debugging.assert_greater_equal(
            X,
            a,
            message="Edges not implemented for Matern52",
            name="assert_left_edge",
        )
        tf.debugging.assert_less_equal(
            X,
            b,
            message="Edges not implemented for Matern52",
            name="assert_right_edge",
        )
    else:
        raise NotImplementedError
    return np.concatenate([Kuf_cos, Kuf_sin], axis=0)


def make_Kuf_np(X, a, b, ms):
    omegas = 2.0 * np.pi * ms / (b - a)
    Kuf_cos = np.cos(omegas * (X - a)).T
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = np.sin(omegas * (X - a)).T
    return np.vstack([Kuf_cos, Kuf_sin])


# ---- kronecker_ops ----

def kron_two(A, B):
    """compute the Kronecker product of two tensorfow tensors"""
    shape = tf.stack([tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]])
    return tf.reshape(
        tf.expand_dims(tf.expand_dims(A, 1), 3) * tf.expand_dims(tf.expand_dims(B, 0), 2), shape
    )


def kron(K):
    return reduce(kron_two, K)


def make_kvs_two(A, B):
    """
    compute the Kronecer-Vector stack of the matrices A and B
    """
    shape = tf.stack([tf.shape(A)[0], tf.shape(A)[1] * tf.shape(B)[1]])
    return tf.reshape(tf.matmul(tf.expand_dims(A, 2), tf.expand_dims(B, 1)), shape)


def make_kvs(k):
    """Compute the kronecker-vector stack of the list of matrices k"""
    return reduce(make_kvs_two, k)


def make_kvs_two_np(A, B):
    # return np.tile(A, [B.shape[0], 1]) * np.repeat(B, A.shape[0], axis=0)
    return np.repeat(A, B.shape[0], axis=0) * np.tile(B, [A.shape[0], 1])


def make_kvs_np(A_list):
    return reduce(make_kvs_two_np, A_list)


# ------- gpr -------

class GPR_kron(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernel_list):

        for kernel in kernel_list:
            assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))

        likelihood = gpflow.likelihoods.Gaussian()
        mean_function = gpflow.mean_functions.Zero()
        num_latent_gps = 1
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = data
        self.X, self.Y = data
        self.a = a
        self.b = b
        # self.ms = ms
        if len(ms) == 1 or isinstance(ms, np.ndarray): # Temporary
            self.ms = [ms[0] for _ in kernel_list]
        else:
            self.ms = ms
        
        assert len(self.ms) == len(kernel_list), "Number of ms must match number of kernels."

        self.input_dim = self.X.shape[1]
        self.kernels = kernel_list

        # count the inducing variables:
        self.Ms = []
        for i, kern in enumerate(kernel_list):
            Ncos_d = self.ms[i].size
            Nsin_d = self.ms[i].size - 1
            self.Ms.append(Ncos_d + Nsin_d)

        # pre compute static quantities
        assert np.all(self.X > a)
        assert np.all(self.X < b)
        Kuf = [
            make_Kuf_np(self.X[:, i : i + 1], a, b, self.ms[i])
            for i, (a, b) in enumerate(zip(self.a, self.b))
        ]
        self.Kuf = make_kvs_np(Kuf)
        self.KufY = np.dot(self.Kuf, self.Y)
        self.KufKfu = np.dot(self.Kuf, self.Kuf.T)
        self.tr_YTY = np.sum(np.square(self.Y))

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        Kdiag = reduce(
            tf.multiply, [k.K_diag(self.X[:, i : i + 1]) for i, k in enumerate(self.kernels)]
        )
        Kuu = [make_Kuu(k, a, b, self.ms[i]) for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))]
        Kuu_solid = kron([Kuu_d.get() for Kuu_d in Kuu])
        Kuu_inv_solid = kron([Kuu_d.inv().get() for Kuu_d in Kuu])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu_solid
        L = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        Kuu_logdets = [K.logdet() for K in Kuu]
        N_others = [float(np.prod(self.Ms[i])) / M for i, M in enumerate(self.Ms)]
        Kuu_logdet = reduce(tf.add, [N * logdet for N, logdet in zip(N_others, Kuu_logdets)])

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), default_float())
        D = tf.cast(tf.shape(self.Y)[1], default_float())
        
        elbo = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        elbo -= 0.5 * D * log_det_P
        elbo += 0.5 * D * Kuu_logdet
        elbo -= 0.5 * self.tr_YTY / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo -= 0.5 * tf.reduce_sum(Kdiag) / sigma2
        elbo += 0.5 * tf.reduce_sum(Kuu_inv_solid * self.KufKfu) / sigma2
        
        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov
        Kuu = [make_Kuu(k, a, b, self.ms[i]) for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))]
        Kuu_solid = kron([Kuu_d.get() for Kuu_d in Kuu])
        Kuu_inv_solid = kron([Kuu_d.inv().get() for Kuu_d in Kuu])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu_solid
        L = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        Kus = [
            make_Kuf(k, Xnew[:, i : i + 1], a, b, self.ms[i])
            for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))
        ]
        Kus = tf.transpose(make_kvs([tf.transpose(Kus_d) for Kus_d in Kus]))
        tmp = tf.linalg.triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = tf.matmul(Kuu_inv_solid, Kus)
        if full_cov:
            raise NotImplementedError
        else:
            var = reduce(
                tf.multiply, [k.K_diag(Xnew[:, i : i + 1]) for i, k in enumerate(self.kernels)]
            )
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var

        