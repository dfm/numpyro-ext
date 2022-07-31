__all__ = ["whiten", "information"]

from functools import singledispatch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from numpyro import distributions as dist
from numpyro import handlers


@singledispatch
def whiten(d):
    raise ValueError(f"Unhandled observed site type: {type(d)}")


@whiten.register(dist.Normal)
def _(d):
    return d.loc / d.scale


@whiten.register(dist.MultivariateNormal)
def _(d):
    return solve_triangular(d.scale_tril, d.loc, lower=True)


try:
    from tinygp.numpyro_support import TinyDistribution
except ImportError:
    pass
else:

    @whiten.register(TinyDistribution)
    def _(d):
        return d.gp.solver.solve_triangular(d.loc)


def information(model, params, *args, invert=False, **kwargs):
    def inner(params, unravel, model, *args, **kwargs):
        trace = handlers.trace(
            handlers.substitute(model, data=unravel(params))
        ).get_trace(*args, **kwargs)

        results = []
        for site in trace.values():
            if site["type"] != "sample" or not site["is_observed"]:
                continue
            results.append(whiten(site["fn"]))

        if not results:
            raise ValueError("No observed sites")

        return tuple(results)

    flat_params, unravel = ravel_pytree(params)
    Js = jax.jacobian(inner)(flat_params, unravel, model, *args, **kwargs)
    F = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))
    for J in Js:
        F += J.T @ J

    if invert:
        F = jnp.linalg.inv(F)

    def unravel_batched(row):
        if jnp.ndim(row) == 1:
            return unravel(row)
        func = unravel
        for n in range(1, jnp.ndim(row)):
            func = jax.vmap(func, in_axes=(n,))
        return func(row)

    return jax.tree_map(unravel_batched, jax.vmap(unravel)(F))
