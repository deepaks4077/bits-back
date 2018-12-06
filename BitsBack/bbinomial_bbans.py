"""
This file contains pop and append method for BBANS using a model inheriting from 
nn.Module from PyTorch.
"""

import torch
import numpy as np
import distributions
import vae_original
import datasets

from numpy.random import RandomState
from numpy_torch_interface import torch_to_numpy_function
from bbans import VAE_append, VAE_pop
from ans import ANSCoder

def build_beta_binomial_bbans(prior_precision, beta_binomial_precision, q_precision, generative_model, recognition_model, n, latent_shape):
    """
    Returns the append and pop functions for mnist bbans implementation
    using Beta-binomial distribution for symbols.
    """

    #append and pop using beta binomial (only thing that changes compared to bernoulli)
    
    def obs_pop(obs_parameters):
        a_list = obs_parameters[0][0]
        b_list = obs_parameters[1][0]
        n_list = [n]*len(a_list)
        return distributions.beta_binomial_obs_pop(beta_binomial_precision)(n_list, a_list, b_list)

    def obs_append(obs_parameters):
        a_list = obs_parameters[0][0]
        b_list = obs_parameters[1][0]
        n_list = [n]*len(a_list)
        return distributions.beta_binomial_obs_append(beta_binomial_precision)(n_list, a_list, b_list)

    #return append, pop for VAE

    append = VAE_append(latent_shape, generative_model, recognition_model,
                        obs_append, prior_precision, q_precision)

    pop = VAE_pop(latent_shape, generative_model, recognition_model,
             obs_pop, prior_precision, q_precision)

    return append, pop