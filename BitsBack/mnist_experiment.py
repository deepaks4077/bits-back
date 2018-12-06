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

def original_bbinomial_example(number_images):
    """
    Run a test session
    """

    rng = RandomState(0)
    image_count = number_images

    prior_precision = 8
    beta_binomial_precision = 14
    q_precision = 14

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

    random_bits = rng.randint(low=0, high=((1 << 32) - 1), size=50, dtype=np.uint32) # some amount of bits 
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