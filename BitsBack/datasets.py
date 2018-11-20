"""
Obtain the MNIST and binarized MNIST datasets
"""

from torchvision import datasets, transforms
import numpy as np
from numpy.random import RandomState

def download_MNIST(use_training_set=False):
    """
    Use Torchvision datasets and transform to a float WxHxD tensor.
    In the case of MNIST, we have a 28x28x1 tensor
    
    Args:
    use_training_set:   If true, downloads the 60 000 images dataset, else downloads the 10 000 images dataset.

    Returns:
    numpy.ndarray of shape (n, 28, 28) with dtype uint8
    """
    mnist = datasets.MNIST('Datasets\\MNIST', train=False, download=True, 
    transform=transforms.Compose([transforms.ToTensor()]))
    if(use_training_set == True):
        return mnist.train_data.numpy()
    else:
        return mnist.test_data.numpy()
    
def get_binarized_MNIST(random_state, use_training_set=False):
    """
    Build a binarized version of MNIST. 
    Process: For each pixel, compute probability by pixel/255. Use random state to sample a random value between 0 and 1. 
    If sample < probablity, set to 0, else set to 1. This ensures that 255 and 0 pixels get mapped to 1 and 0 exactly. 
    All the value in between get mapped to 0 or 1 depending on the sample and the original value.

    Args:
    random_state:       Instance of RandomState class
    use_training_set:   Use the test or training set from MNIST

    Returns:
    numpy.ndarray of shape (n, 28, 28) with dtype bool
    """
    mnist = download_MNIST(use_training_set)
    probabilities = mnist / 255
    b_mnist = random_state.random_sample(np.shape(probabilities)) < probabilities   # returns a numpy array of booleans
    
    return b_mnist


    