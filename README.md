# numpyro-ext


## Usage

```python
>>> import jax
>>> import jax.numpy as jnp

```

### Distributions

The tradition is to import `numpyro_ext.distributions` as `distx` to
differentiate from `numpyro.distributions`, which is imported as `dist`:

```python
>>> from numpyro_ext import distributions as distx

```

**Angle**: A uniform distribution over angles in radians. The actual sampling is
performed in the two-dimensional vector space proportional to `(sin(theta),
cos(theta))` so that the sampler doesn't see a discontinuity at pi.

```python
>>> angle = distx.Angle()
>>> print(angle.sample(jax.random.PRNGKey(0), (2, 3)))
[[ 0.4681001  -2.5152469   1.5203166 ]
 [ 2.4766953   0.6067456  -0.29372737]]

```

**UnitDisk**: A uniform distribution over two-dimensional points within the disk
of radius 1. This means that the sum over squares of the last dimension of a
random variable generated from this distribution will always be less than 1.

```python
>>> unit_disk = distx.UnitDisk()
>>> u = unit_disk.sample(jax.random.PRNGKey(0), (5,))
>>> print(jnp.sum(u**2, axis=-1))
[0.07239353 0.02032685 0.9796876  0.8786792  0.16457272]

```

**NoncentralChi2**: A [non-central chi-squared
distribution](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution).
To use this distribution, you'll need to install the optional
`tensorflow-probability` dependency.

```python
>>> ncx2 = distx.NoncentralChi2(df=3, nc=2.)
>>> print(ncx2.sample(jax.random.PRNGKey(0), (5,)))
[2.197162  3.116601  4.59572   2.8928897 7.2579913]

```
