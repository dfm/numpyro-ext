import jax
from numpyro import handlers, infer


def prior_sample(model, num_samples):
    pred = infer.Predictive(model, num_samples=num_samples)

    def sample(rng_key, *args, **kwargs):
        # Generate samples from the prior
        samples = pred(rng_key, *args, **kwargs)

        # The log likelihood function for a single sample, which we will vmap
        # over the prior samples. Note that we could potentially also use
        # numpyro.infer.util.log_likelihood, but that seems to fail when the
        # model includes "factor" nodes so here we just implement it ourselves.
        def log_like_fn(sample):
            trace = handlers.trace(handlers.substitute(model, sample)).get_trace(
                *args, **kwargs
            )
            result = 0.0
            for site in trace.values():
                if site["type"] == "sample" and site["is_observed"]:
                    result += site["fn"].log_prob(site["value"]).sum()
            return result

        log_like = jax.vmap(log_like_fn)(samples)
        return samples, log_like

    return sample
