"""
Contains experiments on the mnist dataset
"""

import torch
import numpy as np
import distributions
import vae_original
import datasets
from bbinomial_bbans import build_beta_binomial_bbans
from numpy.random import RandomState
from numpy_torch_interface import torch_to_numpy_function
from bbans import VAE_append, VAE_pop
from ans import ANSCoder
from result import Result

def original_bbinomial_example(number_images, result_path:str):
    """
    Run a test session
    """
    seed = 0
    rng = RandomState(seed)
    image_count = number_images

    prior_precision = 8
    beta_binomial_precision = 14
    q_precision = 14

    random_count = 50

    latent_dim = 50
    hidden_dim = 200

    number_buckets = 255 #(trials in the beta-binomial distribution, parameter n. It has to be less than 1 << beta_binomial_precision)

    ans = ANSCoder()

    model = vae_original.BetaBinomialVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load('OriginalParameters\\torch_vae_beta_binomial_params', map_location='cpu'))
    model.eval()

    # obtain numpy compatible functions
    generative_model = torch_to_numpy_function(model.decode)
    recognition_model = torch_to_numpy_function(model.encode)
 
    latent_shape = (1, latent_dim)
    append, pop = build_beta_binomial_bbans(prior_precision, beta_binomial_precision, q_precision, 
                                    generative_model, recognition_model, number_buckets, latent_shape)


    #
    # Get images to compress
    #

    images = datasets.get_binarized_MNIST(rng, False)[:image_count]
    images = [np.atleast_2d(image.flatten()) for image in images]
    original_length = 32 * image_count * 784     #using a float32 per pixel. (not binary)

    # generate some random bits for the

    random_bits = rng.randint(low=0, high=((1 << 32) - 1), size=random_count, dtype=np.uint32) # some amount of bits 
    ans.from_array(random_bits)

    print("Encoding...")
    for i in range(0, len(images)):
        image = images[i]
        append(ans, image)

    compressed_length = 32 * len(ans.to_array())

    print("Decoding...")

    for i in range(len(images) - 1, -1, -1):
        image = pop(ans)
        image = image.detach().numpy()
        original_image = images[i][0]
        assert all(original_image == image)

    # this verifies that all bits from the images have been removed and the ans state is restored
    assert all(ans.to_array() == random_bits)
    print("Decoded all images successfully")


    result = Result()

    result.exp_name = 'Compression of MNIST dataset using BBANS with Beta Binomial distribution'
    result.method_name = 'BBANS using VAE with Beta Binomial latent variables'
    result.path_to_model = 'OriginalParameters/torch_vae_beta_binomial_params'
    result.image_count = image_count
    result.latent_precision = beta_binomial_precision
    result.prior_precision = prior_precision
    result.posterior_precision = q_precision
    result.hidden_size = hidden_dim
    result.latent_size = latent_dim
    result.image_shape = (28, 28)
    result.random_bits_count = random_count * 32
    result.seed = seed
    result.original_length = original_length
    result.compressed_length = compressed_length
    result.encode_success = True
    result.decode_success = True

    result.to_file(result_path)