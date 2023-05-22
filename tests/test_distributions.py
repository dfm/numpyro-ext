import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from scipy.stats import ks_2samp

from numpyro_ext import distributions as distx


def test_log_prob():
    x = jnp.linspace(-1, 1, 100)
    y = jnp.sin(x)
    yerr = 0.1
    design = jnp.vander(x, 2)

    prior = dist.Normal(0.0, 1.0).expand([2])
    data = dist.Normal(jnp.zeros_like(x), yerr)
    full = distx.MarginalizedLinear(jnp.vander(x, 2), prior, data)

    b = jnp.zeros_like(y)
    B = yerr**2 * jnp.eye(len(x)) + design @ jnp.eye(2) @ design.T
    expect = dist.MultivariateNormal(b, B)

    np.testing.assert_allclose(full.log_prob(y), expect.log_prob(y), rtol=1e-5)


def test_numerical_posterior():
    # First sample the full model
    x = jnp.linspace(-1, 1, 100)
    yerr = 0.1

    def model(x, yerr, y=None):
        logs = numpyro.sample("logs", dist.Normal(0.0, 1.0))
        w = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([2]))
        numpyro.sample(
            "y",
            dist.Normal(
                jnp.dot(jnp.vander(x, 2), w), jnp.sqrt(yerr**2 + jnp.exp(2 * logs))
            ),
            obs=y,
        )

    y = infer.Predictive(model, num_samples=1)(jax.random.PRNGKey(0), x, yerr)["y"][0]
    mcmc = infer.MCMC(
        infer.NUTS(model),
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(1), x, yerr, y=y)

    # Then sample the marginalized model
    def marg_model(x, yerr, y=None):
        logs = numpyro.sample("logs", dist.Normal(0.0, 1.0))
        prior = dist.Normal(0.0, 1.0).expand([2])
        data = dist.Normal(jnp.zeros_like(x), jnp.sqrt(yerr**2 + jnp.exp(2 * logs)))
        marg = distx.MarginalizedLinear(jnp.vander(x, 2), prior, data)
        numpyro.sample("y", marg, obs=y)
        if y is not None:
            numpyro.sample("w", marg.conditional(y))

    marg_mcmc = infer.MCMC(
        infer.NUTS(marg_model),
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        progress_bar=False,
    )
    marg_mcmc.run(jax.random.PRNGKey(2), x, yerr, y=y)

    # Check that the results are K-S consistent
    a = mcmc.get_samples()["logs"]
    b = marg_mcmc.get_samples()["logs"]
    np.testing.assert_allclose(jnp.mean(a), jnp.mean(b), rtol=0.01)
    kstest = ks_2samp(a, b)
    assert kstest.pvalue > 0.01

    # Check the conditional distribution
    a = mcmc.get_samples()["w"]
    b = marg_mcmc.get_samples()["w"]
    kstest = ks_2samp(a[:, 0], b[:, 0])
    assert kstest.pvalue > 0.01
    kstest = ks_2samp(a[:, 1], b[:, 1])
    assert kstest.pvalue > 0.01
