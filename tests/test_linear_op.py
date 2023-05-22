import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions
from numpyro_ext.linear_op import to_linear_op


@pytest.mark.parametrize(
    "dist, expected_cov",
    [
        (distributions.Normal(0.0, 1.0), jnp.ones((1, 1))),
        (
            distributions.Normal(jnp.zeros(10), jnp.linspace(0.1, 0.5, 10)),
            jnp.diag(jnp.linspace(0.1, 0.5, 10) ** 2),
        ),
        (
            distributions.Normal(0.0, jnp.linspace(0.1, 0.5, 10)),
            jnp.diag(jnp.linspace(0.1, 0.5, 10) ** 2),
        ),
        (distributions.Normal(jnp.zeros(10), 1.0), jnp.eye(10)),
        (
            distributions.MultivariateNormal(
                jnp.zeros(10), jnp.diag(jnp.linspace(0.1, 0.5, 10))
            ),
            jnp.diag(jnp.linspace(0.1, 0.5, 10)),
        ),
        (
            distributions.MultivariateNormal(0.0, jnp.diag(jnp.linspace(0.1, 0.5, 10))),
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
    np.testing.assert_allclose(op.solve_tril(mu[:, None])[:, 0], alpha)

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
        op.solve_tril(mu[..., None])[..., 0], np.broadcast_to(alpha, shape)
    )
