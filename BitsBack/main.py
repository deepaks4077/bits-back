"""
The main testing coding is launched from here
"""

import datasets
from bin_minst_experiment import original_bernoulli_example
from mnist_experiment import original_bbinomial_example

def main():
    """
    Run examples
    """
    #original_bernoulli_example(10)
    original_bbinomial_example(10)

    

def download_datasets():
    datasets.download_MNIST()

if __name__ == "__main__":
    main()