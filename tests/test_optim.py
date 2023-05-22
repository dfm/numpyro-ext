import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro import distributions

from numpyro_ext import optim


@pytest.mark.parametrize("mu", [0.0, jnp.linspace(-1.0, 1.0, 5)])
def test_optimize(mu):
    def model():
        x = numpyro.sample("x", distributions.Normal(mu, 1.0))
        y = numpyro.sample("y", distributions.Normal(mu, 2.0))
        numpyro.deterministic("sm", x + y)

    soln = optim.optimize(model)(jax.random.PRNGKey(0))
    np.testing.assert_allclose(soln["x"], mu, atol=1e-3)
    np.testing.assert_allclose(soln["y"], mu, atol=1e-3)
    np.testing.assert_allclose(soln["sm"], mu + mu, atol=1e-3)

    soln = optim.optimize(model, ["x"], include_deterministics=False)(
        jax.random.PRNGKey(0)
    )
    np.testing.assert_allclose(soln["x"], mu, atol=1e-3)
    assert not np.allclose(soln["y"], mu, atol=1e-3)
    assert "sm" not in soln

    soln, info = optim.optimize(model, ["y"], return_info=True)(jax.random.PRNGKey(0))
    assert not np.allclose(soln["x"], mu, atol=1e-3)
    np.testing.assert_allclose(soln["y"], mu, atol=1e-3)
    assert info.success
