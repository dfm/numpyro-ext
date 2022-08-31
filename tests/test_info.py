import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro import distributions as dist

from numpyro_ext import info


@pytest.fixture
def linear_data():
    x = jnp.linspace(0, 10, 100)
    y = 3 * x + 2
    K = 1.5 * jnp.exp(
        -0.5 * (x[:, None] - x[None, :]) ** 2 / 0.5**2
    ) + 1e-3 * jnp.eye(len(x))
    A = jnp.vander(x, 2)
    expect = A.T @ jnp.linalg.solve(K, A)
    return x, y, K, expect


def test_linear(linear_data):
    x, y, K, expect = linear_data

    def model(x, y=None):
        A = jnp.vander(x, 2)
        w = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([2]))
        numpyro.sample(
            "y", dist.MultivariateNormal(loc=A @ w, covariance_matrix=K), obs=y
        )

    params = {"w": jnp.zeros(2)}
    calc = info.information(model, params, x, y=y)
    np.testing.assert_allclose(calc["w"]["w"], expect, rtol=2e-6)

    calc = info.information(model, params, x, y=y, invert=True)
    np.testing.assert_allclose(
        calc["w"]["w"], jnp.linalg.inv(expect), rtol=2e-6
    )


def test_linear_multi_in(linear_data):
    x, y, K, expect = linear_data

    def model(x, y=None):
        m = numpyro.sample("m", dist.Normal(0.0, 1.0))
        b = numpyro.sample("b", dist.Normal(0.0, 1.0))
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=m * x + b, covariance_matrix=K),
            obs=y,
        )

    params = {"m": 0.0, "b": 0.0}
    calc = info.information(model, params, x, y=y)
    calc = jnp.array(
        [
            [calc["m"]["m"], calc["m"]["b"]],
            [calc["b"]["m"], calc["b"]["b"]],
        ]
    )
    np.testing.assert_allclose(calc, expect, rtol=2e-6)

    calc = info.information(model, params, x, y=y, invert=True)
    calc = jnp.array(
        [
            [calc["m"]["m"], calc["m"]["b"]],
            [calc["b"]["m"], calc["b"]["b"]],
        ]
    )
    np.testing.assert_allclose(calc, jnp.linalg.inv(expect), rtol=2e-6)


def test_linear_multi_out(linear_data):
    x, y, _, _ = linear_data
    yerr = 1.0
    A = jnp.vander(x, 2)
    expect = A.T @ (A / yerr**2)

    def model(x, y=None):
        A = jnp.vander(x, 2)
        w = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([2]))
        mu = A @ w
        for n in range(len(x)):
            numpyro.sample(f"y{n}", dist.Normal(mu[n], yerr), obs=y[n])

    params = {"w": jnp.zeros(2)}
    calc = info.information(model, params, x, y=y)
    np.testing.assert_allclose(calc["w"]["w"], expect, rtol=2e-6)

    calc = info.information(model, params, x, y=y, invert=True)
    np.testing.assert_allclose(
        calc["w"]["w"], jnp.linalg.inv(expect), rtol=2e-6
    )
