__all__ = ["optimize", "JAXOptMinimize"]

from contextlib import ExitStack

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from numpyro.infer.initialization import init_to_median, init_to_value
from numpyro.optim import _NumPyroOptim


def optimize(
    model, sites=None, start=None, *, init_strategy=None, optimizer=None
):
    if start is not None:
        init_strategy = init_to_value(values=start)
    elif init_strategy is None:
        init_strategy = init_to_median()

    optimizer = JAXOptMinimize() if optimizer is None else optimizer
    guide = AutoDelta(model, sites=sites, init_loc_fn=init_strategy)
    svi = infer.SVI(model, guide, optimizer, loss=infer.Trace_ELBO())

    def run(rng_key, *args, **kwargs):
        init_key, sample_key = jax.random.split(rng_key)
        state = svi.init(init_key, *args, **kwargs)
        state, _ = svi.update(state, *args, **kwargs)
        params = svi.get_params(state)
        return guide.sample_posterior(sample_key, params)

    return run


def _jaxopt_wrapper():
    ident = lambda x: x
    update = lambda *_, x: x
    return ident, update, ident


class JAXOptMinimize(_NumPyroOptim):
    def __init__(self, *args, solver_kwargs=None, **kwargs):
        try:
            import jaxopt
        except ImportError:
            raise ImportError("jaxopt must be installed to use JAXOptMinimize")

        super().__init__(_jaxopt_wrapper)
        self.args = args
        self.kwargs = kwargs
        self.solver_kwargs = {} if solver_kwargs is None else solver_kwargs

    def eval_and_update(self, fn, in_state):
        from jaxopt import ScipyMinimize

        def loss(p):
            out, aux = fn(p, *self.args, **self.kwargs)
            if aux is not None:
                raise ValueError(
                    "JAXOptMinimize does not support models with mutable states."
                )
            return out

        solver = ScipyMinimize(fun=loss, **self.solver_kwargs)
        i, params = in_state
        out_state = solver.run(params)
        return (out_state.state.fun_val, None), (i + 1, out_state.params)


class AutoDelta(infer.autoguide.AutoDelta):
    def __init__(
        self,
        model,
        sites=None,
        *,
        prefix="auto",
        init_loc_fn=infer.init_to_median,
        create_plates=None
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
                    site_fn = dist.Delta(self._init_locs[name]).to_event(
                        event_dim
                    )

                result[name] = numpyro.sample(name, site_fn)

        return result

    def sample_posterior(self, rng_key, params, sample_shape=()):
        del rng_key
        locs = {
            k: params.get("{}_{}_loc".format(k, self.prefix), v)
            for k, v in self._init_locs.items()
        }
        latent_samples = {
            k: jnp.broadcast_to(v, sample_shape + jnp.shape(v))
            for k, v in locs.items()
        }
        return latent_samples
