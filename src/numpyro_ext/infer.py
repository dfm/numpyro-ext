from numpyro import infer


def prior_sample(model, num_samples):
    pred = infer.Predictive(model, num_samples=num_samples)

    def sample(rng_key, *args, **kwargs):
        samples = pred(rng_key, *args, **kwargs)
        log_like = infer.util.log_likelihood(
            model, samples, *args, parallel=True, **kwargs
        )
        return samples, log_like

    return sample
