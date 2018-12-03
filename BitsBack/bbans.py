"""
This class contains the code to encode/decode data using BB-ANS
"""

from ans import ANSCoder

def BBANS_append(posterior_pop, likelihood_append, prior_append):
    """
    Given functions to pop a posterior, append a likelihood and append the prior,
    return a function to append some data.
    """

    def append(ans, data):
        latent = posterior_pop(ans, data)
        likelihood_append(ans, latent, data)
        prior_append(ans, latent)

    return append

def BBANS_pop(prior_pop, likelihood_pop, posterior_append):
    """
    Given functions to pop a prior and likelihood and append
    the posterior, return a function to pop data
    """

    def pop(ans):
        latent = prior_pop(ans)
        data = likelihood_pop(ans, latent)
        posterior_append(ans, latent, data)
        return data
    return pop
