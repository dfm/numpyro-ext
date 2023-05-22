# Extensions for NumPyro

This library includes a miscellaneous set of helper functions, custom
distributions, and other utilities that I find useful when using
[NumPyro](https://num.pyro.ai) in my work.

## Installation

Since NumPyro, and hence this library, are built on top of JAX, it's typically
good practice to start by installing JAX following [the installation
instructions](https://jax.readthedocs.io/en/latest/#installation). Then, you can
install this library using pip:

```bash
python -m pip install numpyro-ext
```

## Usage

Since this README is checked using `doctest`, let's start by importing some
common modules that we'll need in all our examples:

```python
>>> import jax
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro_ext

```

### Distributions

The tradition is to import `numpyro_ext.distributions` as `distx` to
differentiate from `numpyro.distributions`, which is imported as `dist`:

```python
>>> from numpyro import distributions as dist
>>> from numpyro_ext import distributions as distx
>>> key = jax.random.PRNGKey(0)

```

#### Angle

A uniform distribution over angles in radians. The actual sampling is performed
in the two-dimensional vector space proportional to `(sin(theta), cos(theta))`
so that the sampler doesn't see a discontinuity at pi.

```python
>>> angle = distx.Angle()
>>> print(angle.sample(key, (2, 3)))
[[ 0.4...]
 [ 2.4...]]

```

#### UnitDisk

A uniform distribution over two-dimensional points within the disk of radius 1.
This means that the sum over squares of the last dimension of a random variable
generated from this distribution will always be less than 1.

```python
>>> unit_disk = distx.UnitDisk()
>>> u = unit_disk.sample(key, (5,))
>>> print(jnp.sum(u**2, axis=-1))
[0.07...]

```

####  NoncentralChi2

A [non-central chi-squared
distribution](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution).
To use this distribution, you'll need to install the optional
`tensorflow-probability` dependency.

```python
>>> ncx2 = distx.NoncentralChi2(df=3, nc=2.)
>>> print(ncx2.sample(key, (5,)))
[2.19...]

```

#### MarginalizedLinear

The marginalized product of two (possibly multivariate) normal distributions
with a linear relationship between them. The mathematical details of these
models are discussed in detail in [this note](https://arxiv.org/abs/2005.14199),
and this distribution implements the math presented there, in a computationally
efficient way, assuming that the number of marginalized parameters is small
compared to the size of the dataset.

The following example shows a particularly simple example of a
fully-marginalized model for fitting a line to data:

```python
>>> def model(x, y=None):
...     design_matrix = jnp.vander(x, 2)
...     prior = dist.Normal(0.0, 1.0)
...     data = dist.Normal(0.0, 2.0)
...     numpyro.sample(
...         "y",
...         distx.MarginalizedLinear(design_matrix, prior, data),
...         obs=y
...     )
...

```

Things get a little more interesting when the design matrix and/or the
distributions are functions of non-linear parameters. For example, if we want to
find the period of a sinusoidal signal, also fitting for some unknown excess
measurement uncertainty (often called "jitter") we can use the following model:

```python
>>> def model(x, y_err, y=None):
...     period = numpyro.sample("period", dist.Uniform(1.0, 250.0))
...     ln_jitter = numpyro.sample("ln_jitter", dist.Normal(0.0, 2.0))
...     design_matrix = jnp.stack(
...         [
...             jnp.sin(2 * jnp.pi * x / period),
...             jnp.cos(2 * jnp.pi * x / period),
...             jnp.ones_like(x),
...         ],
...         axis=-1,
...     )
...     prior = dist.Normal(0.0, 10.0).expand([3])
...     data = dist.Normal(0.0, jnp.sqrt(y_err**2 + jnp.exp(2*ln_jitter)))
...     numpyro.sample(
...         "y",
...         distx.MarginalizedLinear(design_matrix, prior, data),
...         obs=y
...     )
...
>>> x = jnp.linspace(-1.0, 1.0, 5)
>>> samples = numpyro.infer.Predictive(model, num_samples=2)(key, x, 0.1)
>>> print(samples["period"])
[... ...]
>>> print(samples["y"])
[[... ... ...]
 [... ... ...]]

```

It's often useful to also track conditional samples of the marginalized
parameters during inference. The conditional distribution can be accessed using
the `conditional` method on `MarginalizedLinear`:

```python
>>> x = jnp.linspace(-1.0, 1.0, 5)
>>> y = jnp.sin(x)  # just some fake data
>>> design_matrix = jnp.vander(x, 2)
>>> prior = dist.Normal(0.0, 1.0)
>>> data = dist.Normal(0.0, 2.0)
>>> marg = distx.MarginalizedLinear(design_matrix, prior, data)
>>> cond = marg.conditional(y)
>>> print(type(cond).__name__)
MultivariateNormal
>>> print(cond.sample(key, (3,)))
[[...]
 [...]
 [...]]

```

### Optimization

The inference lore is a little mixed on the benefits of optimization as an
initialization tool for MCMC, but I find that at least in a lot of astronomy
applications, an initial optimization can make a huge difference in performance.
Even if you don't want to use the optimization results as an initialization, it
can still sometimes be useful to numerically search for the maximum _a
posteriori_ parameters for your model. However, the NumPyro interface for these
types of optimization isn't terribly user-friendly, so this library provides
some helpers to make it a little more straightforward.

By default, this optimization uses the wrappers of scipy's optimization routines
provided by the [JAXopt](https://github.com/google/jaxopt) library, so you'll
need to install JAXopt:

```bash
python -m pip install jaxopt
```

before running these examples.

The following example shows a simple optimization of a model with a single
parameter:

```python
>>> from numpyro_ext import optim as optimx
>>>
>>> def model(y=None):
...     x = numpyro.sample("x", dist.Normal(0.0, 1.0))
...     numpyro.sample("y", dist.Normal(x, 2.0), obs=y)
...
>>> soln = optimx.optimize(model)(key, y=0.5)

```

By default, the optimization starts from a prior sample, but you can provide
custom initial coordinates as follows:

```python
>>> soln = optimx.optimize(model, start={"x": 12.3})(key, y=0.5)

```

Similarly, if you only want to optimize a subset of the parameters, you can
provide a list of parameters to target:

```python
>>> soln = optimx.optimize(model, sites=["x"])(key, y=0.5)

```

### Information matrix computation

The Fisher information matrix for models with Gaussian likelihoods is
[straightforward to
compute](https://en.wikipedia.org/wiki/Fisher_information#Multivariate_normal_distribution),
and this library provides a helper function for automating this computation:

```python
>>> from numpyro_ext import information
>>>
>>> def model(x, y=None):
...     a = numpyro.sample("a", dist.Normal(0.0, 1.0))
...     b = numpyro.sample("b", dist.Normal(0.0, 1.0))
...     log_alpha = numpyro.sample("log_alpha", dist.Normal(0.0, 1.0))
...     cov = jnp.exp(log_alpha - 0.5 * (x[:, None] - x[None, :])**2)
...     cov += 0.1 * jnp.eye(len(x))
...     numpyro.sample(
...         "y",
...         dist.MultivariateNormal(loc=a * x + b, covariance_matrix=cov),
...         obs=y,
...     )
...
>>> x = jnp.linspace(-1.0, 1.0, 5)
>>> y = jnp.sin(x)  # the input data just needs to have the right shape
>>> params = {"a": 0.5, "b": -0.2, "log_alpha": -0.5}
>>> info = information(model)(params, x, y=y)
>>> print(info)
{'a': {'a': ..., 'b': ... 'log_alpha': ...}, 'b': ...}

```

The returned information matrix is a nested dictionary of dictionaries, indexed
by pairs of parameter names, where the values are the corresponding blocks of
the information matrix.
