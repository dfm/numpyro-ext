import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro import distributions as dist

from numpyro_ext import info


def assert_allclose(a, b, **kwargs):
    kwargs["rtol"] = kwargs.get("rtol", 2e-5)
    return np.testing.assert_allclose(a, b, **kwargs)


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
    calc = info.information(model)(params, x, y=y)
    assert_allclose(calc["w"]["w"], expect)

    calc = info.information(model, invert=True)(params, x, y=y)
    assert_allclose(calc["w"]["w"], jnp.linalg.inv(expect))


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
    calc = info.information(model)(params, x, y=y)
    calc = jnp.array(
        [
            [calc["m"]["m"], calc["m"]["b"]],
            [calc["b"]["m"], calc["b"]["b"]],
        ]
    )
    assert_allclose(calc, expect)

    calc = info.information(model, invert=True)(params, x, y=y)
    calc = jnp.array(
        [
            [calc["m"]["m"], calc["m"]["b"]],
            [calc["b"]["m"], calc["b"]["b"]],
        ]
    )
    assert_allclose(calc, jnp.linalg.inv(expect))


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
    calc = info.information(model)(params, x, y=y)
    assert_allclose(calc["w"]["w"], expect)

    calc = info.information(model, invert=True)(params, x, y=y)
    assert_allclose(calc["w"]["w"], jnp.linalg.inv(expect))


def test_factor():
    def rosenbrock(x, y, a=1.0, b=100.0):
        return jnp.log(jnp.square(a - x) + b * jnp.square(y - x**2))

    def model():
        x = numpyro.sample("x", dist.Uniform(-2, 2))
        y = numpyro.sample("y", dist.Uniform(-1, 3))
        numpyro.factor("prior", rosenbrock(x, y))

    params = {"x": 0.0, "y": 2.0}
    expect = jax.hessian(rosenbrock, argnums=(0, 1))(params["x"], params["y"])
    calc = info.information(model, include_prior=True)(params)
    assert_allclose(calc["x"]["x"], -expect[0][0])
    assert_allclose(calc["x"]["y"], -expect[0][1])
    assert_allclose(calc["y"]["x"], -expect[1][0])
    assert_allclose(calc["y"]["y"], -expect[1][1])


def test_unconstrained():
    def model1(y=None):
        x_ = numpyro.sample(
            "x", dist.ImproperUniform(dist.constraints.real, (), event_shape=(1,))
        )
        x = dist.transforms.SigmoidTransform()(x_)
        numpyro.sample("y", dist.Normal(x, 1.0), obs=y)

    def model2(y=None):
        x = numpyro.sample("x", dist.Uniform(0.0, 1.0))
        numpyro.sample("y", dist.Normal(x, 1.0), obs=y)

    y = 0.1
    info1 = info.information(model1)({"x": 0.1}, y=y)
    info2 = info.information(model2, unconstrained=True)({"x": 0.1}, y=y)
    assert_allclose(info1["x"]["x"], info2["x"]["x"])
