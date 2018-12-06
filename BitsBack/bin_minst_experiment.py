"""
Contains experiments on the binarized mnist dataset
"""
import torch
import numpy as np
import distributions
import bin_vae_original
import datasets
from result import Result
from bernoulli_bbans import build_bernoulli_bbans
from numpy.random import RandomState
from numpy_torch_interface import torch_to_numpy_function
from bbans import VAE_append, VAE_pop
from ans import ANSCoder
import BinaryVAE

# original experiment from the paper
def original_bernoulli_example(number_images, result_path:str):
    """
    Test the binary mnist on VAE BBANS 
    """
    seed = 0

    rng = RandomState(seed)
    image_count = number_images

    prior_precision = 8
    bernoulli_precision = 12
    q_precision = 14
    random_count = 12
    latent_dim = 40
    hidden_dim = 100

    ans = ANSCoder()
    result = Result()

    model = bin_vae_original.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load('OriginalParameters/torch_binary_vae_params_new'))

    generative_model = torch_to_numpy_function(model.decode)
    recognition_model = torch_to_numpy_function(model.encode)

    latent_shape = (latent_dim,)

    append, pop = build_bernoulli_bbans(prior_precision, bernoulli_precision, q_precision, generative_model, recognition_model, latent_shape)


    # Get images to compress

    images = datasets.get_binarized_MNIST(rng, False)[:image_count]
    images = [image.flatten() for image in images]
    original_length = 32 * len(images) * len(images[0])     #using a float32 per pixel. Could be optimized to 8 bits per pixel.

    # generate some random bits

    random_bits = rng.randint(low=0, high=((1 << 32) - 1), size=random_count, dtype=np.uint32) 
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
        original_image = images[i]
        assert all(original_image == image)

    # this verifies that all bits from the images have been removed and the ans state is restored
    assert all(ans.to_array() == random_bits)

    print("All images  decoded successfully!")


    result = Result()

    result.exp_name = 'Compression of binarized MNIST dataset using BBANS with Bernoulli distribution'
    result.method_name = 'BBANS using VAE with Bernoulli latent variables'
    result.path_to_model = 'OriginalParameters/torch_binary_vae_params_new'
    result.image_count = image_count
    result.latent_precision = bernoulli_precision
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

# New VAEs experiments:

def bernoulli_100_100_40_20_False_5(number_images, result_path:str):
    path = 'Parameters/bb_binary_vae_100_100_40_20_False_5'
    latent_dim = 40
    hidden_dim = 100
    model = BinaryVAE.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    bernoulli_experiment(number_images, result_path, latent_dim, hidden_dim, model, path)

def bernoulli_100_100_20_20_False_3(number_images, result_path:str):
    path = 'Parameters/binary_vae_100_100_20_20_False_3'
    latent_dim = 20
    hidden_dim = 100
    model = BinaryVAE.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    bernoulli_experiment(number_images, result_path, latent_dim, hidden_dim, model, path)

def bernoulli_100_150_20_20_False_5(number_images, result_path:str):
    path = 'Parameters/binary_vae_100_150_20_20_False_5'
    latent_dim = 20
    hidden_dim = 150
    model = BinaryVAE.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    bernoulli_experiment(number_images, result_path, latent_dim, hidden_dim, model, path)

def bernoulli_100_150_40_20_False_5(number_images, result_path:str):
    path = 'Parameters/binary_vae_100_150_40_20_False_5'
    latent_dim = 40
    hidden_dim = 150
    model = BinaryVAE.BinaryVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    bernoulli_experiment(number_images, result_path, latent_dim, hidden_dim, model, path)

def bernoulli_experiment(number_images, result_path, latent_dim, hidden_dim, model, path):
    """
    Run an experiment given the model
    """
    seed = 0
    random_count = 12

    generative_model = torch_to_numpy_function(model.decode)
    recognition_model = torch_to_numpy_function(model.encode)

    rng = RandomState(seed)
    image_count = number_images

    prior_precision = 8
    bernoulli_precision = 12
    q_precision = 14


    latent_shape = (latent_dim,)
    ans = ANSCoder()
    result = Result()

    append, pop = build_bernoulli_bbans(prior_precision, bernoulli_precision, q_precision, generative_model, recognition_model, latent_shape)

    # Get images to compress

    images = datasets.get_binarized_MNIST(rng, False)[:image_count]
    images = [image.flatten() for image in images]
    original_length = 32 * len(images) * len(images[0])     #using a float32 per pixel. Could be optimized to 8 bits per pixel.

    # generate some random bits

    random_bits = rng.randint(low=0, high=((1 << 32) - 1), size=random_count, dtype=np.uint32) 
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
        original_image = images[i]
        assert all(original_image == image)

    # this verifies that all bits from the images have been removed and the ans state is restored
    assert all(ans.to_array() == random_bits)

    print("All images  decoded successfully!")


    result = Result()

    result.exp_name = 'Compression of binarized MNIST dataset using BBANS with Bernoulli distribution'
    result.method_name = 'BBANS using VAE with Bernoulli latent variables'
    result.path_to_model = path
    result.image_count = image_count
    result.latent_precision = bernoulli_precision
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
    