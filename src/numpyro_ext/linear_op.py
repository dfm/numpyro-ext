from functools import singledispatch
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_solve, solve_triangular
from numpyro.distributions import ExpandedDistribution, MultivariateNormal, Normal

if hasattr(jax, "Array"):
    Array = jax.Array
else:
    Array = Any


class LinearOp(NamedTuple):
    loc: Callable[[], Array]
    covariance: Callable[[], Array]
    inverse: Callable[[], Array]
    solve_tril: Callable[[Array, bool], Array]
    half_log_det: Callable[[], Array]


@singledispatch
def to_linear_op(dist) -> LinearOp:
    raise ValueError(f"{type(dist)} doesn't support the 'to_linear_op' interface")


@to_linear_op.register(Normal)
def _(dist):
    scale = jnp.broadcast_to(dist.scale, dist.shape())

    def loc():
        return jnp.broadcast_to(dist.loc, dist.shape())

    def covariance():
        return jnp.vectorize(jnp.diag, signature="(n)->(n,n)")(
            jnp.square(jnp.atleast_1d(scale))
        )

    def inverse():
        return jnp.vectorize(jnp.diag, signature="(n)->(n,n)")(
            1.0 / jnp.square(jnp.atleast_1d(scale))
        )

    def solve_tril(y, transpose):
        del transpose
        return y / jnp.atleast_1d(scale)[..., None]

    def half_log_det():
        return jnp.sum(jnp.log(jnp.atleast_1d(scale)), axis=-1)

    return LinearOp(loc, covariance, inverse, solve_tril, half_log_det)


@to_linear_op.register(MultivariateNormal)
def _(dist):
    def loc():
        return jnp.broadcast_to(dist.loc, dist.shape())

    def covariance():
        return dist.covariance_matrix

    def inverse():
        y = jnp.broadcast_to(
            jnp.eye(dist.scale_tril.shape[-1]), dist.covariance_matrix.shape
        )
        return cho_solve((dist.scale_tril, True), y)

    def solve_tril(y, transpose):
        return solve_triangular(dist.scale_tril, y, trans=transpose, lower=True)

    def half_log_det():
        return jnp.sum(jnp.log(jnp.diagonal(dist.scale_tril, axis1=-2, axis2=-1)), -1)

    return LinearOp(loc, covariance, inverse, solve_tril, half_log_det)


@to_linear_op.register(ExpandedDistribution)
def _(dist):
    (
        base_loc,
        base_covariance,
        base_inverse,
        base_solve_tril,
        base_half_log_det,
    ) = to_linear_op(dist.base_dist)
    shape = dist.batch_shape + dist.event_shape
    batch_shape = shape[:-1]
    event_shape = shape[-1:]

    def loc():
        mu = base_loc()
        return jnp.broadcast_to(mu, shape)

    def covariance():
        cov = base_covariance()
        if cov.shape[-1:] != event_shape:
            assert cov.shape[-1] == 1
            cov = jnp.eye(event_shape[0]) * cov
        return jnp.broadcast_to(cov, batch_shape + event_shape + event_shape)

    def inverse():
        inv = base_inverse()
        if inv.shape[-1:] != event_shape:
            assert inv.shape[-1] == 1
            inv = jnp.eye(event_shape[0]) * inv
        return jnp.broadcast_to(inv, batch_shape + event_shape + event_shape)

    def solve_tril(y, transpose):
        if jnp.ndim(y) < 2:
            raise ValueError(
                "An expanded linear operator's inverse is only defined for matrices"
            )

        shape = lax.broadcast_shapes(
            batch_shape,
            jnp.shape(y)[: max(jnp.ndim(y) - 2, 0)],
        )
        alpha = jnp.vectorize(
            lambda x: base_solve_tril(x, transpose), signature="(m,k)->(m,k)"
        )(y)
        return jnp.broadcast_to(alpha, shape + y.shape[-2:])

    def half_log_det():
        hld = base_half_log_det()
        # Special case for scalar base distribution
        if dist.base_dist.shape() == ():
            hld *= event_shape[0]
        return jnp.broadcast_to(hld, batch_shape)

    return LinearOp(loc, covariance, inverse, solve_tril, half_log_det)
