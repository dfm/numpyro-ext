__all__ = ["information"]

from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpyro import handlers, infer

from numpyro_ext.linear_op import to_linear_op


def standardize(d):
    op = to_linear_op(d)
    return op.solve_tril(op.loc(), False)


def _is_conditioned(site):
    return site["is_observed"] and not site["infer"].get("is_auxiliary", False)


def _information_and_log_prior_hessian(
    model,
    params,
    model_args=(),
    model_kwargs=None,
    invert=False,
    include_prior=False,
    unconstrained=False,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs

    # Determine which parameters are sampled
    trace = handlers.trace(handlers.substitute(model, data=params)).get_trace(
        *model_args, **model_kwargs
    )
    base_params = {}
    for site in trace.values():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        if site["name"] not in params:
            raise KeyError(f"Input params is missing the site called '{site['name']}'")
        base_params[site["name"]] = params[site["name"]]

    # This function computes the terms of the likelihood that we will
    # differentiate to get the information matrix, and the log prior
    def impl(params, unravel, model, model_args, model_kwargs):
        params = unravel(params)
        if unconstrained:
            substituted_model = handlers.substitute(
                model,
                substitute_fn=partial(infer.util._unconstrain_reparam, params),
            )
        else:
            substituted_model = handlers.substitute(model, data=params)

        trace = handlers.trace(substituted_model).get_trace(*model_args, **model_kwargs)

        info_terms = []
        log_prior = jnp.zeros(())
        for site in trace.values():
            if site["type"] != "sample":
                continue

            # If a site is observed, we need to include it in the information
            # computation, but some sites will be labeled as observed when
            # they're actually the Jacobians of transforms. In these cases, we
            # want to include them in the prior instead.
            if _is_conditioned(site):
                info_terms.append(standardize(site["fn"]))
            else:
                log_prior += jnp.sum(site["fn"].log_prob(site["value"]))

        return tuple(info_terms) or (0.0,), log_prior

    flat_params, unravel = ravel_pytree(base_params)

    # Compute the Jacobian of the model to evaluate the information matrix
    Js = jax.jacobian(lambda *args: impl(*args)[0])(
        flat_params, unravel, model, model_args, model_kwargs
    )
    F = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))
    for J in Js:
        F += jnp.einsum("...n,...m->nm", J, J)

    # Compute the Hessian of the log prior function
    if include_prior:
        H = jax.hessian(lambda *args: impl(*args)[1])(
            flat_params, unravel, model, model_args, model_kwargs
        )

        # Combine the two
        F = F - H

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


def information(model, invert=False, include_prior=False, unconstrained=False):
    """Compute the Fisher information matrix for a NumPyro model

    Note that this only supports a limited set of observation sites. By default,
    this requires either ``Normal`` or ``MultivariateNormal`` distributions for
    observed sites, but custom distributions can be supported by registering a
    custom ``standardize`` transformation. Take a look at the source code for
    ``numpyro_ext.info.standardize`` for some examples.

    Args:
        model: The NumPyro model definition.
        invert: If ``True``, the inverse information matrix will be returned.
        include_prior: If ``True``, the Hessian of the log prior will be
            subtracted from the information matrix.
        unconstrained: If ``True``, the parameters are assumed to be in the
            unconstrained space and the information is computed in that space.

    Returns:
        A callable with the signature ``def info(params, *args, **kwargs)`` to
        compute the information matrix, where ``params`` is a dictionary of the
        parameters where the information will be computed, and the other
        arguments are the static arguments for ``model``.

    """

    return lambda params, *args, **kwargs: _information_and_log_prior_hessian(
        model,
        params,
        model_args=args,
        model_kwargs=kwargs,
        invert=invert,
        include_prior=include_prior,
        unconstrained=unconstrained,
    )
