__all__ = ["optimize", "JAXOptMinimize"]

from contextlib import ExitStack

import jax
import jax.numpy as jnp
import numpyro
from jax.tree_util import tree_map
from numpyro import distributions as dist
from numpyro import infer
from numpyro.infer.initialization import init_to_median, init_to_value
from numpyro.optim import _NumPyroOptim


def optimize(
    model,
    sites=None,
    start=None,
    *,
    init_strategy=None,
    optimizer=None,
    num_steps=1,
    include_deterministics=True,
    return_info=False,
):
    """Numerically maximize the log probability of a NumPyro model

    The main feature that this interface supports is that it enables optimizing
    a subset of the parameters in an automated fashion, something which can be
    tricky with the built in NumPyro functions.

    Example:

    .. code-block:: python

        def model(x, yerr, y=None):
            A = jnp.vander(x, 2)
            w = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([2]))
            mu = numpyro.deterministic("mu", A @ w)
            numpyro.sample("y", dist.Normal(mu, yerr), obs=y)

        run_optim = optim.optimize(model)
        param = run_optim(jax.random.PRNGKey(0), x, yerr, y=y)

    Args:
        model: The NumPyro model definition.
        sites: A list of the site names to vary, keeping the others fixed. By
            default, all parameters are varied.
        start: A dictionary of initial site values keyed by site name. For
            sites not included in ``sites``, this will be the fixed value used
            for that site.
        init_strategy: If ``start`` is provided, this will be ignored.
            Otherwise, this specifies the initial values for the sites in the
            optimization. By default, this take the value ``init_to_median``.
        optimizer: A NumPyro optimizer object to use as the optimization engine.
            By default this uses a ``JAXOptMinimize`` optimizer.
        num_steps: The number of optimization steps to run. The default
            ``JAXOptMinimize`` optimizer only requires one step, so this is the
            default.
        include_deterministics: If ``True``, return the values of the
            deterministics computed at the optimized parameters, in addition to
            the parameter values.
        return_info: If ``True``, the returned function will return a tuple with
            the parameters as the first element, and scipy's minimization status
            as the second element.

    Returns:
        A callable that will execute the optimization routine, with the
        signature ``run(random_key, *args, **kwargs)`` where ``random_key`` is a
        ``jax.random.PRNGKey``, and ``*args`` and ``**kwargs`` are the static
        arguments for ``model``.
    """
    if start is not None:
        init_strategy = init_to_value(values=start)
    elif init_strategy is None:
        init_strategy = init_to_median()

    optimizer = JAXOptMinimize() if optimizer is None else optimizer
    guide = AutoDelta(model, sites=sites, init_loc_fn=init_strategy)
    svi = infer.SVI(model, guide, optimizer, loss=infer.Trace_ELBO())

    def run(rng_key, *args, **kwargs):
        init_key, sample_key, pred_key = jax.random.split(rng_key, 3)
        state = svi.init(init_key, *args, **kwargs)
        for _ in range(num_steps):
            state, _ = svi.update(state, *args, **kwargs)
        info = getattr(state.optim_state[1], "state", None)
        params = svi.get_params(state)
        sample = guide.sample_posterior(sample_key, params)
        if include_deterministics:
            pred = tree_map(
                lambda x: x[0],
                infer.Predictive(model, tree_map(lambda x: x[None], sample))(
                    pred_key, *args, **kwargs
                ),
            )
            sample = dict(sample, **pred)
        if return_info:
            return sample, info
        return sample

    return run


def _jaxopt_wrapper():
    def init_fn(params):
        from jaxopt._src.scipy_wrappers import ScipyMinimizeInfo
        from jaxopt.base import OptStep

        return OptStep(
            params=params,
            state=ScipyMinimizeInfo(
                fun_val=jnp.zeros(()),
                success=False,
                status=0,
                iter_num=0,
                hess_inv=None,
            ),
        )

    def update_fn(i, grad_tree, opt_state):
        return opt_state

    def get_params_fn(opt_state):
        return opt_state.params

    return init_fn, update_fn, get_params_fn


class JAXOptMinimize(_NumPyroOptim):
    """A NumPyro-compatible optimizer built using jaxopt.ScipyMinimize

    This exposes the ``ScipyMinimize`` optimizer from ``jaxopt`` to NumPyro. All
    keyword arguments are passed directly to ``jaxopt.ScipyMinimize``.
    """

    def __init__(self, **kwargs):
        try:
            pass
        except ImportError as e:
            raise ImportError("jaxopt must be installed to use JAXOptMinimize") from e

        super().__init__(_jaxopt_wrapper)
        self.solver_kwargs = {} if kwargs is None else kwargs

    def eval_and_update(self, fn, in_state, forward_mode_differentiation=False):
        import scipy.optimize  # noqa
        from jaxopt import ScipyMinimize

        def loss(p):
            out, aux = fn(p)
            if aux is not None:
                raise ValueError(
                    "JAXOptMinimize does not support models with mutable states."
                )
            return out

        if forward_mode_differentiation:
            raise ValueError(
                "Forward mode differentiation is not implemented for JaxOptMinimze"
            )

        solver = ScipyMinimize(fun=loss, **self.solver_kwargs)
        out_state = solver.run(self.get_params(in_state))
        return (out_state.state.fun_val, None), (in_state[0] + 1, out_state)


class AutoDelta(infer.autoguide.AutoDelta):
    """A MAP autoguide with support for keeping some sites fixed

    This is an extension of ``numpyro.infer.autoguide.AutoDelta`` that adds
    support for only varying a subset of sites. All arguments except for
    ``sites`` are passed directly to the upstream implementation.

    Args:
        sites: A list of the site names to vary, keeping the others fixed. By
            default, all parameters are varied.
    """

    def __init__(
        self,
        model,
        sites=None,
        *,
        prefix="auto",
        init_loc_fn=infer.init_to_median,
        create_plates=None,
    ):
        self._sites = sites
        super().__init__(
            model,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            create_plates=create_plates,
        )

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = self._event_dims[name]
            init_loc = self._init_locs[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                if self._sites is None or name in self._sites:
                    site_loc = numpyro.param(
                        "{}_{}_loc".format(name, self.prefix),
                        init_loc,
                        constraint=site["fn"].support,
                        event_dim=event_dim,
                    )

                    site_fn = dist.Delta(site_loc).to_event(event_dim)

                else:
                    site_fn = dist.Delta(self._init_locs[name]).to_event(event_dim)

                result[name] = numpyro.sample(name, site_fn)

        return result

    def sample_posterior(self, rng_key, params, sample_shape=()):
        del rng_key
        locs = {
            k: params.get("{}_{}_loc".format(k, self.prefix), v)
            for k, v in self._init_locs.items()
        }
        latent_samples = {
            k: jnp.broadcast_to(v, sample_shape + jnp.shape(v)) for k, v in locs.items()
        }
        return latent_samples
