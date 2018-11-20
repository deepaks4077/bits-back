"""
The main testing coding is launched from here
"""
import numpy as np
from numpy.random import RandomState


import datasets

def main():
    random_state = RandomState(123)

    mnist = datasets.download_MNIST(use_training_set=False)
    b_mnist = datasets.get_binarized_MNIST(random_state, use_training_set=False)

    






if __name__ == "__main__":
    main()