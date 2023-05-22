import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro import distributions as dist

from numpyro_ext import infer as inferx


@pytest.mark.parametrize("prior", [dist.Normal(0.0, 1.0), dist.Uniform(0.0, 1.0)])
def test_prior_sample(prior):
    def model(y=None):
        x = numpyro.sample("x", prior)
        numpyro.sample("y", dist.Normal(x, 2.0), obs=y)

    samples, log_like = inferx.prior_sample(model, 100)(jax.random.PRNGKey(0), y=1.5)
    expect = -0.5 * ((samples["x"] - samples["y"]) / 2.0) ** 2 - 0.5 * jnp.log(
        2 * jnp.pi * 2.0**2
    )
    np.testing.assert_allclose(log_like, expect)
