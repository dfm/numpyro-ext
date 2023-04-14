from functools import singledispatch

import jax.numpy as jnp
from jax import lax, random
from jax.scipy.linalg import cho_factor, cho_solve
from numpyro.distributions import (
    ExpandedDistribution,
    MultivariateNormal,
    Normal,
    constraints,
)
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key, validate_sample


@singledispatch
def to_linear_op(dist):
    raise ValueError(
        f"{type(dist)} doesn't support the 'to_linear_op' interface"
    )


@to_linear_op.register(Normal)
def _(dist):
    def covariance():
        return jnp.vectorize(jnp.diag, signature="(n)->(n,n)")(
            jnp.square(jnp.atleast_1d(dist.scale))
        )

    def apply_inverse(y):
        return y / jnp.square(dist.scale)[..., None]

    def half_log_det():
        return jnp.sum(jnp.log(jnp.atleast_1d(dist.scale)), axis=-1)

    return covariance, apply_inverse, half_log_det


@to_linear_op.register(MultivariateNormal)
def _(dist):
    def covariance():
        return dist.covariance_matrix

    def apply_inverse(y):
        return cho_solve((dist.scale_tril, True), y)

    def half_log_det():
        return jnp.sum(
            jnp.log(jnp.diagonal(dist.scale_tril, axis1=-2, axis2=-1)), -1
        )

    return covariance, apply_inverse, half_log_det


@to_linear_op.register(ExpandedDistribution)
def _(dist):
    base_covariance, base_apply_inverse, base_half_log_det = to_linear_op(
        dist.base_dist
    )
    shape = dist.batch_shape + dist.event_shape
    batch_shape = shape[:-1]
    event_shape = shape[-1:]

    def covariance():
        cov = base_covariance()
        if cov.shape[-1:] != event_shape:
            assert cov.shape[-1] == 1
            cov = jnp.eye(event_shape[0]) * cov
        return jnp.broadcast_to(cov, batch_shape + event_shape + event_shape)

    def apply_inverse(y):
        if jnp.ndim(y) < 2:
            raise ValueError(
                "A linear operator's inverse is only defined for matrices"
            )
        shape = lax.broadcast_shapes(
            batch_shape,
            jnp.shape(y)[: max(jnp.ndim(y) - 2, 0)],
        )
        return jnp.broadcast_to(base_apply_inverse(y), shape + y.shape[-2:])

    def half_log_det():
        hld = base_half_log_det()
        # Special case for scalar base distribution
        if dist.base_dist.shape() == ():
            hld *= event_shape[0]
        return jnp.broadcast_to(hld, batch_shape)

    return covariance, apply_inverse, half_log_det


class MarginalizedLinear(Distribution):
    arg_constraints = {"design_matrix": constraints.real}
    support = constraints.real_vector
    reparametrized_params = ["design_matrix"]

    def __init__(
        self,
        design_matrix,
        prior_distribution,
        data_distribution,
        *,
        validate_args=None,
    ):
        # We treat the trailing dimensions of the design matrix as "ground
        # truth" for the dimensions of the problem.
        if jnp.ndim(design_matrix) < 2:
            raise ValueError(
                "The design matrix must have at least 2 dimensions"
            )
        data_size, latent_size = jnp.shape(design_matrix)[-2:]

        # We don't really care about the batch vs. event shapes of the input
        # distributions, so instead we just check that the trailing dimensions
        # are correct.
        prior_dist_shape = tuple(prior_distribution.batch_shape) + tuple(
            prior_distribution.event_shape
        )
        if len(prior_dist_shape) != 0 and prior_dist_shape[-1] != latent_size:
            raise ValueError(
                "The trailing dimensions of the prior distribution must match "
                "the latent dimension defined by the design matrix; expected "
                f"(..., {latent_size}), got {prior_dist_shape}"
            )

        data_dist_shape = tuple(data_distribution.batch_shape) + tuple(
            data_distribution.event_shape
        )
        if len(data_dist_shape) != 0 and data_dist_shape[-1] != data_size:
            raise ValueError(
                "The trailing dimensions of the data distribution must match "
                "the data dimension defined by the design matrix; expected "
                f"(..., {data_size}), got {data_dist_shape}"
            )

        # We broadcast the relevant batch shapes to find the batch shape of this
        # distribution, and expand or reshape all the members to match.
        batch_shape = lax.broadcast_shapes(
            design_matrix.shape[:-2],
            prior_dist_shape[:-1],
            data_dist_shape[:-1],
        )
        event_shape = (data_size,)
        self.design_matrix = jnp.broadcast_to(
            design_matrix, batch_shape + (data_size, latent_size)
        )

        if prior_distribution.event_shape == ():
            self.prior_distribution = prior_distribution.expand(
                batch_shape + (latent_size,)
            )
        else:
            self.prior_distribution = prior_distribution.expand(batch_shape)

        if data_distribution.event_shape == ():
            self.data_distribution = data_distribution.expand(
                batch_shape + (data_size,)
            )
        else:
            self.data_distribution = data_distribution.expand(batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        prior_key, data_key = random.split(key)
        prior_sample = self.prior_distribution.sample(
            prior_key, sample_shape=sample_shape
        )
        data_sample = self.data_distribution.sample(
            data_key, sample_shape=sample_shape
        )
        delta = jnp.einsum(
            "...ij,...j->...i", self.design_matrix, prior_sample
        )
        return data_sample + delta

    @validate_sample
    def log_prob(self, value):
        data_size, latent_size = jnp.shape(self.design_matrix)[-2:]
        assert jnp.shape(value)[-1] == data_size
        _, prior_solve, prior_hld = to_linear_op(self.prior_distribution)
        _, data_solve, data_hld = to_linear_op(self.data_distribution)
        design_matrix_T = jnp.swapaxes(self.design_matrix, -2, -1)

        # This inner matrix is used for both the matrix determinant and inverse
        sigma = prior_solve(
            jnp.eye(latent_size)
        ) + design_matrix_T @ data_solve(self.design_matrix)
        factor, lower = cho_factor(sigma, lower=True)

        # Use the matrix determinant lemma to compute the full determinant
        hld = jnp.sum(jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)), -1)
        norm = hld + prior_hld() + data_hld()

        # Use the Woodbury matrix identity to solve the linear system
        resid = (value - self.mean)[..., None]
        alpha = data_solve(resid)
        result = jnp.swapaxes(resid, -2, -1) @ alpha
        alpha = design_matrix_T @ alpha
        result -= jnp.swapaxes(alpha, -2, -1) @ cho_solve(
            (factor, lower), alpha
        )

        return (
            -0.5 * result[..., 0, 0]
            - norm
            - 0.5 * data_size * jnp.log(2 * jnp.pi)
        )

    def tree_flatten(self):
        prior_flat, prior_aux = self.prior_distribution.tree_flatten()
        data_flat, data_aux = self.data_distribution.tree_flatten()
        return (self.design_matrix, prior_flat, data_flat), (
            type(self.prior_distribution),
            prior_aux,
            type(self.data_distribution),
            data_aux,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        design_matrix, prior_flat, data_flat = params
        prior_dist = aux_data[0].tree_unflatten(aux_data[1], prior_flat)
        data_dist = aux_data[2].tree_unflatten(aux_data[3], data_flat)
        return cls(
            design_matrix=design_matrix,
            prior_distribution=prior_dist,
            data_distribution=data_dist,
        )

    @property
    def mean(self):
        mu = self.prior_distribution.mean[..., None]
        mu = jnp.broadcast_to(mu, self.batch_shape + mu.shape[-2:])
        return self.data_distribution.mean + (self.design_matrix @ mu)[..., 0]

    @property
    def covariance_matrix(self):
        prior_cov, _, _ = to_linear_op(self.prior_distribution)
        data_cov, _, _ = to_linear_op(self.data_distribution)
        return data_cov() + self.design_matrix @ prior_cov() @ jnp.swapaxes(
            self.design_matrix, -2, -1
        )
