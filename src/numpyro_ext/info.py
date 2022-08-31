__all__ = ["information", "standardize"]

from functools import singledispatch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from numpyro import distributions as dist
from numpyro import handlers


def information(model, params, *args, invert=False, **kwargs):
    """Compute the Fisher information matrix for a NumPyro model

    Note that this only supports a limited set of observation sites. By default,
    this requires either ``Normal`` or ``MultivariateNormal`` distributions for
    observed sites, but custom distributions can be supported by registering a
    custom ``standardize`` transformation. Take a look at the source code for
    ``numpyro_ext.info.standardize`` for some examples.

    All extra arguments and keyword arguments are passed as static arguments to
    ``model``.

    Args:
        model: The NumPyro model definition.
        params: A dictionary of the parameters where the information will be
            computed.
        invert: If ``True``, the inverse information matrix will be returned.
    """

    def inner(params, unravel, model, *args, **kwargs):
        trace = handlers.trace(
            handlers.substitute(model, data=unravel(params))
        ).get_trace(*args, **kwargs)

        results = []
        for site in trace.values():
            if site["type"] != "sample" or not site["is_observed"]:
                continue
            results.append(standardize(site["fn"]))

        if not results:
            raise ValueError("No observed sites")

        return tuple(results)

    # Determine which parameters are sampled; there may well be a better way...
    trace = handlers.trace(handlers.substitute(model, data=params)).get_trace(
        *args, **kwargs
    )
    base_params = {}
    for site in trace.values():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        if site["name"] not in params:
            raise KeyError(
                f"Input params is missing the site called '{site['name']}'"
            )
        base_params[site["name"]] = params[site["name"]]

    # Compute the Jacobian
    flat_params, unravel = ravel_pytree(base_params)
    Js = jax.jacobian(inner)(flat_params, unravel, model, *args, **kwargs)
    F = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))
    for J in Js:
        F += jnp.einsum("...n,...m->nm", J, J)

    if invert:
        F = jnp.linalg.inv(F)

    def unravel_batched(row):
        if jnp.ndim(row) == 1:
            return unravel(row)
        func = unravel
        for n in range(1, jnp.ndim(row)):
            func = jax.vmap(func, in_axes=(n,))
        return func(row)

    return jax.tree_util.tree_map(unravel_batched, jax.vmap(unravel)(F))


@singledispatch
def standardize(d):
    raise ValueError(
        f"Unhandled observed site type: {type(d)}\n"
        "To implement information support for this distribution, register "
        "a custom 'standardize' transform:\n"
        f"@numpyro_ext.info.standardize.register({type(d)})\n"
        "def custom(d): ..."
    )


@standardize.register(dist.Normal)
def _(d):
    return d.loc / d.scale


@standardize.register(dist.MultivariateNormal)
def _(d):
    return solve_triangular(d.scale_tril, d.loc, lower=True)


try:
    from tinygp.numpyro_support import TinyDistribution
except ImportError:
    pass
else:

    @standardize.register(TinyDistribution)
    def _(d):
        return d.gp.solver.solve_triangular(d.loc)
