"""
This class contains the code to encode/decode data using BB-ANS
"""

from ans import ANSCoder
import numpy as np
import distributions

def BBANS_append(posterior_pop, likelihood_append, prior_append):
    """
    Given functions to pop a posterior, append a likelihood and append the prior,
    return a function to append some data.
    """

    def append(ans, data):
        latent = posterior_pop(data)(ans)
        likelihood_append(latent)(ans, data)
        prior_append(ans, latent)

    return append

def BBANS_pop(prior_pop, likelihood_pop, posterior_append):
    """
    Given functions to pop a prior and likelihood and append
    the posterior, return a function to pop data
    """

    def pop(ans):
        latent = prior_pop(ans)
        data = likelihood_pop(latent)(ans)
        posterior_append(data)(ans, latent)
        return data
    return pop

def VAE_append(latent_shape, generative_model, recognition_model, 
                obs_append, prior_precision, latent_precision):
    """
    This append takes functions from a variational autoencoder and produces
    an append and pop function for BBANS.
    Follows the same layout as vae_append in the author's code
    """

    def posterior_pop(data):
        """
        Pop the posterior 
        """
        posterior_mean, posterior_stdd = recognition_model(data)
        posterior_mean = np.ravel(posterior_mean)
        posterior_stdd = np.ravel(posterior_stdd)
        # we now have an array of mean and standard deviation values
        # from the posterior distribution

        cdfs = [distributions.gaussian_latent_cdf(mean, stdd, prior_precision, latent_precision) 
                for mean, stdd in zip(posterior_mean, posterior_stdd)]

        ppfs = [distributions.gaussian_latent_ppf(mean, stdd, prior_precision, latent_precision)
                for mean, stdd in zip(posterior_mean, posterior_stdd)]

        return distributions.distr_pop(latent_precision, ppfs, cdfs)

    def likelihood_append(latent_indices):
        """
        Append the likelihood
        """
        y = distributions.standard_gaussian_centers(prior_precision)[latent_indices]
        obs_parameters = generative_model(np.reshape(y, latent_shape))
        return obs_append(obs_parameters)

    prior_append = distributions.uniforms_append(prior_precision)
    return BBANS_append(posterior_pop, likelihood_append, prior_append)

def VAE_pop(latent_shape, generative_model, recognition_model,
             obs_pop, prior_precision, latent_precision):
    """
    Pop a symbol using VAE BBANS
    """
    prior_pop = distributions.uniforms_pop(prior_precision, np.prod(latent_shape))

    def likelihood_pop(latent_indices):
        y = distributions.standard_gaussian_centers(prior_precision)[latent_indices]
        obs_params = generative_model(np.reshape(y, latent_shape))
        return obs_pop(obs_params)

    def posterior_append(data):
        posterior_mean, posterior_stdd = recognition_model(np.atleast_2d(data))
        posterior_mean = np.ravel(posterior_mean)
        posterior_stdd = np.ravel(posterior_stdd)
        cdfs = [distributions.gaussian_latent_cdf(mean, stdd, prior_precision, latent_precision)
                for mean, stdd in zip(posterior_mean, posterior_stdd)]
        return distributions.distr_append(latent_precision, cdfs)

    return BBANS_pop(prior_pop, likelihood_pop, posterior_append)