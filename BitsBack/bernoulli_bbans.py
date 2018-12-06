"""
This file contains pop and append method for BBANS using a model inheriting from 
nn.Module from PyTorch.
"""

import torch
import numpy as np
import distributions
import bin_vae_original
import datasets

from numpy.random import RandomState
from numpy_torch_interface import torch_to_numpy_function
from bbans import VAE_append, VAE_pop
from ans import ANSCoder

def build_bernoulli_bbans(prior_precision, bernoulli_precision, q_precision, generative_model, recognition_model, latent_shape):
    """
    Returns the append and pop functions for bbans using Bernoulli distribution.
    """

    #define append and pop for bernoulli probabilities
    obs_append = distributions.bernoulli_obs_append(bernoulli_precision)
    obs_pop = distributions.bernoulli_obs_pop(bernoulli_precision)

    append = VAE_append(latent_shape, generative_model, recognition_model,
                        obs_append, prior_precision, q_precision)

    pop = VAE_pop(latent_shape, generative_model, recognition_model,
             obs_pop, prior_precision, q_precision)

    return append, pop
