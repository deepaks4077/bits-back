"""
This file contains pop and append method for BBANS using a model inheriting from 
nn.Module from PyTorch.
"""

import torch
import numpy as np
from numpy.random import RandomState

from numpy_torch_interface import torch_to_numpy_function

from bbans import VAE_append, VAE_pop
from ans import ANSCoder
import distributions
import bin_vae_original
import datasets


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
                        obs_append, prior_precision, q_precision)

    pop = VAE_append(latent_shape, generative_model, recognition_model,
             obs_pop, prior_precision, q_precision)

    return append, pop






if __name__ == '__main__':
    """
    Test the binary mnist on VAE BBANS 
    """

    prior_precision = 8
    bernoulli_precision = 12
    q_precision = 14

    latent_dim = 40
    hidden_dim = 100

    ans = ANSCoder()

    model = bin_vae_original.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)

    path_to_params = 'OriginalParameters/torch_binary_vae_params_new'

    append, pop = build_append_pop(prior_precision, bernoulli_precision, q_precision, hidden_dim, latent_dim, model, path_to_params)


    #
    # Get images to compress
    #

    rng = RandomState(0)
    image_count = 100

    images = datasets.get_binarized_MNIST(rng, False)[:image_count]
    images = [image.flatten() for image in images]
    original_length = 32 * len(images) * len(images[0])     #using a float32 per pixel. Could be optimized to 8 bits per pixel.

    # generate some random bits for the

    other_bits = rng.randint(low=0, high=((1 << 32) - 1), size=12, dtype=np.uint32) # total of 640 bits
    ans.from_array(other_bits)


    for i in range(0, len(images)):
        image = images[i]
        append(ans, image)
        print("Completed an image")

    compressed_length = 32 * len(ans.to_array())
    
    bits_per_pixel = compressed_length / (784 * float(image_count))

    print("Original length: " + str(original_length))
    print('Compressed length: ' + str(compressed_length))
    print('Bits per pixel: ' + str(bits_per_pixel))
