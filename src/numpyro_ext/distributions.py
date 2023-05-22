__all__ = ["QuadLDParams", "UnitDisk", "Angle", "MixtureGeneral", "NoncentralChi2"]

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro.distributions as dist
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from numpyro.distributions import MixtureGeneral as MixtureGeneral
from numpyro.distributions import constraints, transforms
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample

from numpyro_ext.linear_op import to_linear_op

# ---------------
# - Constraints -
# ---------------


class _QuadLDConstraint(constraints.Constraint):
    event_dim = 1

    def __call__(self, u):
        assert jnp.shape(u) == (2,)
        a = u[0] + u[1] < 1.0
        b = u[0] > 0.0
        c = u[0] + 2 * u[1] > 0.0
        return a & b & c

    def feasible_like(self, prototype):
        assert jnp.shape(prototype)[-1] == 2
        return QuadLDParams.q2u(jnp.full_like(prototype, 0.5))


class _UnitDiskConstraint(constraints.Constraint):
    event_dim = 1

    def __call__(self, x):
        assert jnp.shape(x) == (2,)
        return x[0] ** 2 + x[1] ** 2 <= 1.0

    def feasible_like(self, prototype):
        assert jnp.shape(prototype)[-1] == 2
        return jnp.zeros_like(prototype)


class _AngleConstraint(constraints._Interval):
    def __init__(self, regularized=10.0):
        self.regularized = regularized
        super().__init__(-jnp.pi, jnp.pi)


quad_ld = _QuadLDConstraint()
unit_disk = _UnitDiskConstraint()
angle = _AngleConstraint()

# --------------
# - Transforms -
# --------------


class QuadLDTransform(transforms.Transform):
    domain = constraints.independent(constraints.unit_interval, 1)
    codomain = quad_ld

    def __eq__(self, other):
        return isinstance(other, QuadLDTransform)

    def __call__(self, q):
        return QuadLDParams.q2u(q)

    def _inverse(self, u):
        return QuadLDParams.u2q(u)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del y, intermediates
        return jnp.zeros_like(x[..., 0])


class UnitDiskTransform(transforms.Transform):
    domain = constraints.independent(constraints.interval(-1.0, 1.0), 1)
    codomain = unit_disk

    def __eq__(self, other):
        return isinstance(other, UnitDiskTransform)

    def __call__(self, x):
        assert jnp.ndim(x) >= 1 and jnp.shape(x)[-1] == 2
        return jnp.stack(
            (
                x[..., 0],
                x[..., 1] * jnp.sqrt(1 - jnp.clip(x[..., 0], -1, 1) ** 2),
            ),
            axis=-1,
        )

    def _inverse(self, y):
        assert jnp.ndim(y) >= 1 and jnp.shape(y)[-1] == 2
        return jnp.stack(
            (
                y[..., 0],
                y[..., 1] / jnp.sqrt(1 - jnp.clip(y[..., 0], -1.0, 1.0) ** 2),
            ),
            axis=-1,
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del y, intermediates
        return 0.5 * jnp.log(1 - jnp.clip(x[..., 0], -1.0, 1.0) ** 2)


class Arctan2Transform(transforms.Transform):
    domain = constraints.real_vector
    codomain = angle

    def __init__(self, regularized=None):
        self.regularized = regularized

    def __eq__(self, other):
        return isinstance(other, Arctan2Transform)

    def __call__(self, x):
        assert jnp.ndim(x) >= 1 and jnp.shape(x)[-1] == 2
        return jnp.arctan2(x[..., 0], x[..., 1])

    def _inverse(self, y):
        return jnp.stack((jnp.sin(y), jnp.cos(y)), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del y, intermediates
        sm = jnp.sum(jnp.square(x), axis=-1)
        if self.regularized is None:
            return -0.5 * sm
        return self.regularized * jnp.log(sm) - 0.5 * sm

    def forward_shape(self, shape):
        return shape[:-1]

    def inverse_shape(self, shape):
        return shape + (2,)


@transforms.biject_to.register(quad_ld)
def _(constraint):
    del constraint
    return transforms.ComposeTransform(
        [
            transforms.SigmoidTransform(),
            QuadLDTransform(),
        ]
    )


@transforms.biject_to.register(unit_disk)
def _(constraint):
    del constraint
    return transforms.ComposeTransform(
        [
            transforms.SigmoidTransform(),
            transforms.AffineTransform(-1.0, 2.0, domain=constraints.unit_interval),
            UnitDiskTransform(),
        ]
    )


@transforms.biject_to.register(angle)
def _(constraint):
    return Arctan2Transform(regularized=constraint.regularized)


# -----------------
# - Distributions -
# -----------------


class QuadLDParams(dist.Distribution):
    support = quad_ld

    def __init__(self, *, validate_args=None):
        super().__init__(batch_shape=(), event_shape=(2,), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        q = jax.random.uniform(key, shape=sample_shape + (2,), minval=0, maxval=1)
        return QuadLDParams.q2u(q)

    @validate_sample
    def log_prob(self, value):
        return jnp.zeros_like(value[..., 0])

    @property
    def mean(self):
        return jnp.array([2.0 / 3.0, 0.0])

    @property
    def variance(self):
        return jnp.array([2.0 / 9.0, 1.0 / 6.0])

    @staticmethod
    def q2u(q):
        assert jnp.ndim(q) >= 1 and jnp.shape(q)[-1] == 2
        q1 = jnp.sqrt(q[..., 0])
        q2 = 2 * q[..., 1]
        u1 = q1 * q2
        u2 = q1 * (1 - q2)
        return jnp.stack((u1, u2), axis=-1)

    @staticmethod
    def u2q(u):
        assert jnp.ndim(u) >= 1 and jnp.shape(u)[-1] == 2
        u1 = u[..., 0]
        u2 = u1 + u[..., 1]
        q1 = jnp.square(u2)
        q2 = 0.5 * u1 / u2
        return jnp.stack((q1, q2), axis=-1)


class UnitDisk(dist.Distribution):
    support = unit_disk

    def __init__(self, *, validate_args=None):
        super().__init__(batch_shape=(), event_shape=(2,), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key1, key2 = jax.random.split(key)
        theta = jax.random.uniform(
            key1, shape=sample_shape, minval=-jnp.pi, maxval=jnp.pi
        )
        r = jnp.sqrt(
            jax.random.uniform(key2, shape=sample_shape, minval=0.0, maxval=1.0)
        )
        return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), axis=-1)

    @validate_sample
    def log_prob(self, value):
        del value
        return -jnp.log(jnp.pi)

    @property
    def mean(self):
        return jnp.array([0.0, 0.0])

    @property
    def variance(self):
        return jnp.array([0.25, 0.25])


class Angle(dist.Distribution):
    def __init__(self, *, regularized=10.0, validate_args=None):
        self.regularized = regularized
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return _AngleConstraint(self.regularized)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return jax.random.uniform(
            key, shape=sample_shape, minval=-jnp.pi, maxval=jnp.pi
        )

    @validate_sample
    def log_prob(self, value):
        del value
        return -jnp.log(jnp.pi)

    @property
    def mean(self):
        return 0.0

    @property
    def variance(self):
        return jnp.pi**2 / 12.0

    def cdf(self, value):
        cdf = (value + 0.5 * jnp.pi) / jnp.pi
        return jnp.clip(cdf, a_min=0.0, a_max=1.0)

    def icdf(self, value):
        return (value - 0.5) * jnp.pi


class NoncentralChi2(dist.Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "nc": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["df", "nc"]

    def __init__(self, df, nc, validate_args=None):
        self.df, self.nc = promote_shapes(df, nc)
        batch_shape = lax.broadcast_shapes(jnp.shape(df), jnp.shape(nc))
        super(NoncentralChi2, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        # Ref: https://github.com/numpy/numpy/blob/
        # 89c80ba606f4346f8df2a31cfcc0e967045a68ed/numpy/
        # random/src/distributions/distributions.c#L797-L813

        def _random_chi2(key, df, shape=(), dtype=jnp.float_):
            return 2.0 * jax.random.gamma(key, 0.5 * df, shape=shape, dtype=dtype)

        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape

        key1, key2, key3 = jax.random.split(key, 3)
        i = jax.random.poisson(key1, 0.5 * self.nc, shape=shape)
        n = jax.random.normal(key2, shape=shape) + jnp.sqrt(self.nc)
        cond = jnp.greater(self.df, 1.0)
        chi2 = _random_chi2(
            key3,
            jnp.where(cond, self.df - 1.0, self.df + 2.0 * i),
            shape=shape,
        )
        return jnp.where(cond, chi2 + n * n, chi2)

    @validate_sample
    def log_prob(self, value):
        try:
            import tensorflow_probability.substrates.jax as tfp
        except ImportError as e:
            raise ImportError(
                "tensorflow-probability is must be installed to use the "
                "NoncentralChi2 distribution."
            ) from e

        # Ref: https://github.com/scipy/scipy/blob/
        # 500878e88eacddc7edba93dda7d9ee5f784e50e6/scipy/
        # stats/_distn_infrastructure.py#L597-L610
        df2 = self.df / 2.0 - 1.0
        xs, ns = jnp.sqrt(value), jnp.sqrt(self.nc)
        res = jsp.special.xlogy(df2 / 2.0, value / self.nc) - 0.5 * (xs - ns) ** 2
        corr = tfp.math.bessel_ive(df2, xs * ns) / 2.0
        return jnp.where(
            jnp.greater(corr, 0.0),
            res + jnp.log(corr),
            -jnp.inf,
        )

    @property
    def mean(self):
        return self.df + self.nc

    @property
    def variance(self):
        return 2.0 * (self.df + 2.0 * self.nc)


class MarginalizedLinear(dist.Distribution):
    arg_constraints = {"design_matrix": constraints.real}
    support = constraints.real_vector
    reparametrized_params = ["design_matrix"]

    def __init__(
        self,
        design_matrix,
        prior_distribution,
        data_distribution,
        *,
        validate_args=None,
    ):
        # We treat the trailing dimensions of the design matrix as "ground
        # truth" for the dimensions of the problem.
        if jnp.ndim(design_matrix) < 2:
            raise ValueError("The design matrix must have at least 2 dimensions")
        data_size, latent_size = jnp.shape(design_matrix)[-2:]

        # We don't really care about the batch vs. event shapes of the input
        # distributions, so instead we just check that the trailing dimensions
        # are correct.
        prior_dist_shape = tuple(prior_distribution.batch_shape) + tuple(
            prior_distribution.event_shape
        )
        if len(prior_dist_shape) != 0 and prior_dist_shape[-1] != latent_size:
            raise ValueError(
                "The trailing dimensions of the prior distribution must match "
                "the latent dimension defined by the design matrix; expected "
                f"(..., {latent_size}), got {prior_dist_shape}"
            )

        data_dist_shape = tuple(data_distribution.batch_shape) + tuple(
            data_distribution.event_shape
        )
        if len(data_dist_shape) != 0 and data_dist_shape[-1] != data_size:
            raise ValueError(
                "The trailing dimensions of the data distribution must match "
                "the data dimension defined by the design matrix; expected "
                f"(..., {data_size}), got {data_dist_shape}"
            )

        # We broadcast the relevant batch shapes to find the batch shape of this
        # distribution, and expand or reshape all the members to match.
        batch_shape = lax.broadcast_shapes(
            design_matrix.shape[:-2],
            prior_dist_shape[:-1],
            data_dist_shape[:-1],
        )
        event_shape = (data_size,)
        self.design_matrix = jnp.broadcast_to(
            design_matrix, batch_shape + (data_size, latent_size)
        )

        if prior_distribution.event_shape == ():
            self.prior_distribution = prior_distribution.expand(
                batch_shape + (latent_size,)
            )
        else:
            self.prior_distribution = prior_distribution.expand(batch_shape)

        if data_distribution.event_shape == ():
            self.data_distribution = data_distribution.expand(
                batch_shape + (data_size,)
            )
        else:
            self.data_distribution = data_distribution.expand(batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample_with_intermediates(self, key, sample_shape=()):
        assert is_prng_key(key)
        prior_key, data_key = jax.random.split(key)
        prior_sample = self.prior_distribution.sample(
            prior_key, sample_shape=sample_shape
        )
        data_sample = self.data_distribution.sample(data_key, sample_shape=sample_shape)
        delta = jnp.einsum("...ij,...j->...i", self.design_matrix, prior_sample)
        return data_sample + delta, [prior_sample]

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key, sample_shape)[0]

    def _solve(self, value):
        data_size, _ = jnp.shape(self.design_matrix)[-2:]
        assert jnp.shape(value)[-1] == data_size
        prior = to_linear_op(self.prior_distribution)
        data = to_linear_op(self.data_distribution)

        def inner(a, b):
            aT = jnp.swapaxes(a, -2, -1)
            return aT @ b

        # This inner matrix is used for both the matrix determinant and inverse
        data_tril = data.solve_tril(self.design_matrix, False)
        sigma = prior.inverse() + inner(data_tril, data_tril)
        factor, lower = cho_factor(sigma, lower=True)

        # Use the matrix determinant lemma to compute the full determinant
        hld = jnp.sum(jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)), -1)
        norm = hld + prior.half_log_det() + data.half_log_det()

        # Use the Woodbury matrix identity to solve the linear system
        resid = (value - self.mean)[..., None]
        alpha = data.solve_tril(resid, False)
        result = inner(alpha, alpha)
        alpha = inner(data_tril, alpha)
        result -= inner(alpha, cho_solve((factor, lower), alpha))
        log_prob = (
            -0.5 * result[..., 0, 0] - norm - 0.5 * data_size * jnp.log(2 * jnp.pi)
        )

        # Evaluate the conditional mean
        a = prior.solve_tril(prior.solve_tril(prior.loc()[..., None], False), True)
        a = cho_solve((factor, lower), (a + alpha)[..., 0])

        return log_prob, a, sigma

    @validate_sample
    def log_prob(self, value):
        return self._solve(value)[0]

    def tree_flatten(self):
        prior_flat, prior_aux = self.prior_distribution.tree_flatten()
        data_flat, data_aux = self.data_distribution.tree_flatten()
        return (self.design_matrix, prior_flat, data_flat), (
            type(self.prior_distribution),
            prior_aux,
            type(self.data_distribution),
            data_aux,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        design_matrix, prior_flat, data_flat = params
        prior_dist = aux_data[0].tree_unflatten(aux_data[1], prior_flat)
        data_dist = aux_data[2].tree_unflatten(aux_data[3], data_flat)
        return cls(
            design_matrix=design_matrix,
            prior_distribution=prior_dist,
            data_distribution=data_dist,
        )

    @property
    def mean(self):
        mu = self.prior_distribution.mean[..., None]
        mu = jnp.broadcast_to(mu, self.batch_shape + mu.shape[-2:])
        return self.data_distribution.mean + (self.design_matrix @ mu)[..., 0]

    @property
    def covariance_matrix(self):
        prior = to_linear_op(self.prior_distribution)
        data = to_linear_op(self.data_distribution)
        return data.cov() + self.design_matrix @ prior.cov() @ jnp.swapaxes(
            self.design_matrix, -2, -1
        )

    def conditional_weights_distribution(self, value):
        _, a, A_inv = self._solve(value)
        return dist.MultivariateNormal(loc=a, precision_matrix=A_inv)
