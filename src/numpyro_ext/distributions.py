__all__ = ["QuadLDParams"]

import warnings

from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.util import find_stack_level
from numpyro.distributions.util import validate_sample, is_prng_key


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
            transforms.AffineTransform(
                -1.0, 2.0, domain=constraints.unit_interval
            ),
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
        super().__init__(
            batch_shape=(), event_shape=(2,), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        q = random.uniform(key, shape=sample_shape + (2,), minval=0, maxval=1)
        return QuadLDParams.q2u(q)

    @validate_sample
    def log_prob(self, value):
        return jnp.zeros_like(value[..., 0])

    # def __init__(self, *, validate_args=None):
    #     base_dist = dist.Uniform().expand([2])
    #     transform = QuadLDTransform()
    #     super().__init__(base_dist, transform, validate_args=validate_args)

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
        super().__init__(
            batch_shape=(), event_shape=(2,), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key1, key2 = random.split(key)
        theta = random.uniform(
            key1, shape=sample_shape, minval=-jnp.pi, maxval=jnp.pi
        )
        r = jnp.sqrt(
            random.uniform(key2, shape=sample_shape, minval=0.0, maxval=1.0)
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
        super().__init__(
            batch_shape=(), event_shape=(), validate_args=validate_args
        )

    @property
    def support(self):
        return _AngleConstraint(self.regularized)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return random.uniform(
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
