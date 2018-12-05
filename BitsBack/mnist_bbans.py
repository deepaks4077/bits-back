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
import vae_original
import datasets


def build_append_pop(prior_precision, beta_binomial_precision, q_precision, hidden_dim, latent_dim, model, n, path_to_params:str):
    """
    Returns the append and pop functions for mnist bbans implementation
    using Beta-binomial distribution for symbols.
    """
    # load trained model
    model.load_state_dict(torch.load(path_to_params, map_location='cpu'))
    model.eval()

    # obtain numpy compatible functions
    generative_model = torch_to_numpy_function(model.decode)
    recognition_model = torch_to_numpy_function(model.encode)

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

    latent_shape = (1, latent_dim)

    append = VAE_append(latent_shape, generative_model, recognition_model,
                        obs_append, prior_precision, q_precision)

    pop = VAE_pop(latent_shape, generative_model, recognition_model,
             obs_pop, prior_precision, q_precision)

    return append, pop



if __name__ == '__main__':
    """
    Run a test session
    """

    

    prior_precision = 8
    beta_binomial_precision = 14
    q_precision = 14

    latent_dim = 50
    hidden_dim = 200

    number_buckets = 255 #(trials in the beta-binomial distribution, parameter n. It has to be less than)

    ans = ANSCoder()

    model = vae_original.BetaBinomialVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)

    path_to_params = 'OriginalParameters\\torch_vae_beta_binomial_params'

    append, pop = build_append_pop(prior_precision, beta_binomial_precision, q_precision, hidden_dim, latent_dim, model, number_buckets, path_to_params)


    #
    # Get images to compress
    #

    rng = RandomState(0)
    image_count = 1

    images = datasets.get_binarized_MNIST(rng, False)[:image_count]
    images = [np.atleast_2d(image.flatten()) for image in images]
    original_length = 32 * image_count * 784     #using a float32 per pixel. (not binary)

    # generate some random bits for the

    random_bits = rng.randint(low=0, high=((1 << 32) - 1), size=50, dtype=np.uint32) # total of 640 bits
    ans.from_array(random_bits)

    print("Encoding...")
    for i in range(0, len(images)):
        image = images[i]
        append(ans, image)

    compressed_length = 32 * len(ans.to_array())

    bits_per_pixel = compressed_length / (784 * float(image_count))

    print("Original length: " + str(original_length))
    print('Compressed length: ' + str(compressed_length))
    print('Bits per pixel: ' + str(bits_per_pixel))

    print("Decoding...")

    for i in range(len(images) - 1, -1, -1):
        image = pop(ans)
        image = image.detach().numpy()
        original_image = images[i][0]
        assert all(original_image == image)

    # this verifies that all bits from the images have been removed and the ans state is restored
    assert all(ans.to_array() == random_bits)
    print("Decoded all images successfully")