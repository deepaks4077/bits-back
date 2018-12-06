"""
Contains experiments on the binarized mnist dataset
"""
import torch
import numpy as np
import distributions
import bin_vae_original
import datasets
from bernoulli_bbans import build_bernoulli_bbans
from numpy.random import RandomState
from numpy_torch_interface import torch_to_numpy_function
from bbans import VAE_append, VAE_pop
from ans import ANSCoder

def original_bernoulli_example(number_images):
    """
    Test the binary mnist on VAE BBANS 
    """

    rng = RandomState(0)
    image_count = number_images

    prior_precision = 8
    bernoulli_precision = 12
    q_precision = 14

    latent_dim = 40
    hidden_dim = 100

    ans = ANSCoder()

    model = bin_vae_original.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load('OriginalParameters/torch_binary_vae_params_new'))

    generative_model = torch_to_numpy_function(model.decode)
    recognition_model = torch_to_numpy_function(model.encode)

    latent_shape = (latent_dim,)

    append, pop = build_bernoulli_bbans(prior_precision, bernoulli_precision, q_precision, generative_model, recognition_model, latent_shape)


    #
    # Get images to compress
    #

    images = datasets.get_binarized_MNIST(rng, False)[:image_count]
    images = [image.flatten() for image in images]
    original_length = 32 * len(images) * len(images[0])     #using a float32 per pixel. Could be optimized to 8 bits per pixel.

    # generate some random bits for the

    random_bits = rng.randint(low=0, high=((1 << 32) - 1), size=12, dtype=np.uint32) # total of 640 bits
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
        original_image = images[i]
        assert all(original_image == image)

    # this verifies that all bits from the images have been removed and the ans state is restored
    assert all(ans.to_array() == random_bits)
    print("Decoded all images successfully")