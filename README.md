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
