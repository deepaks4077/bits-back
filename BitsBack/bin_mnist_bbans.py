"""
This file contains pop and append method for BBANS using a model inheriting from 
nn.Module from PyTorch.
"""

import torch
import numpy as np

from numpy_torch_interface import torch_to_numpy_function

from bbans import VAE_append, VAE_pop
from ans import ANSCoder
import distributions

"""
import util
import rans
from tvae_binary import BinaryVAE
import tvae_utils
from torchvision import datasets, transforms
from torch.distributions import Bernoulli
import time
"""

def build_append_pop(prior_precision, bernoulli_precision, q_precision, hidden_dim, latent_dim, model, path_to_params:str):
    """
    Returns the append and pop functions for a binary mnist bbans implementation
    using Bernoulli distribution for the symbols and pixels
    """
    # load trained model
    model.load_state_dict(torch.load(path_to_params))

    # obtain numpy compatible functions
    generative_model = torch_to_numpy_function(model.decode)
    recognition_model = torch_to_numpy_function(model.encode)

    #append and pop using bernoulli
    obs_append = distributions.bernoulli_obs_append(bernoulli_precision)
    obs_pop = distributions.bernoulli_obs_pop(bernoulli_precision)

    #return append, pop for VAE

    latent_shape = (latent_dim,)

    append = VAE_append(latent_shape, generative_model, recognition_model,
                        obs_append, prior_precision, latent_precision)

    pop = VAE_append(latent_shape, generative_model, recognition_model,
             obs_pop, prior_precision, latent_precision)

    return append, pop
