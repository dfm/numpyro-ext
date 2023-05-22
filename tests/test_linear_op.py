import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions

from numpyro_ext.linear_op import to_linear_op


@pytest.mark.parametrize(
    "dist, expected_cov",
    [
        (distributions.Normal(0.1, 0.5), 0.5**2 * jnp.ones((1, 1))),
        (
            distributions.Normal(0.1 * jnp.ones(10), jnp.linspace(0.1, 0.5, 10)),
            jnp.diag(jnp.linspace(0.1, 0.5, 10) ** 2),
        ),
        (
            distributions.Normal(0.1, jnp.linspace(0.1, 0.5, 10)),
            jnp.diag(jnp.linspace(0.1, 0.5, 10) ** 2),
        ),
        (distributions.Normal(0.1 * jnp.ones(10), 0.5), 0.5**2 * jnp.eye(10)),
        (
            distributions.MultivariateNormal(
                0.1 * jnp.ones(10), jnp.diag(jnp.linspace(0.1, 0.5, 10))
            ),
            jnp.diag(jnp.linspace(0.1, 0.5, 10)),
        ),
        (
            distributions.MultivariateNormal(0.1, jnp.diag(jnp.linspace(0.1, 0.5, 10))),
            jnp.diag(jnp.linspace(0.1, 0.5, 10)),
        ),
    ],
)
def test_linear_op(dist, expected_cov):
    event_shape = expected_cov.shape[:1]
    mu = np.broadcast_to(dist.loc, event_shape)
    alpha = np.linalg.solve(np.linalg.cholesky(expected_cov), mu[:, None])[:, 0]
    expected_inv = np.linalg.inv(expected_cov)
    op = to_linear_op(dist)
    np.testing.assert_allclose(op.loc(), mu)
    np.testing.assert_allclose(op.covariance(), expected_cov)
    np.testing.assert_allclose(op.inverse(), expected_inv, rtol=5e-6)
    np.testing.assert_allclose(
        op.solve_tril(mu[:, None], False)[:, 0], alpha, rtol=5e-6
    )
    np.testing.assert_allclose(
        op.half_log_det(), 0.5 * np.linalg.slogdet(expected_cov)[1]
    )

    shape = [2, 3, 4] + list(event_shape)
    if dist.event_shape:
        exp = dist.expand(shape[:-1])
    else:
        exp = dist.expand(shape)
    mu = np.broadcast_to(mu, shape)
    op = to_linear_op(exp)
    np.testing.assert_allclose(op.loc(), mu)
    np.testing.assert_allclose(
        op.covariance(), np.broadcast_to(expected_cov, shape + list(event_shape))
    )
    np.testing.assert_allclose(
        op.inverse(),
        np.broadcast_to(expected_inv, shape + list(event_shape)),
        rtol=5e-6,
    )
    np.testing.assert_allclose(
        op.solve_tril(mu[..., None], False)[..., 0],
        np.broadcast_to(alpha, shape),
        rtol=5e-6,
    )
    np.testing.assert_allclose(
        op.half_log_det(),
        jnp.broadcast_to(0.5 * np.linalg.slogdet(expected_cov)[1], shape[:-1]),
    )
