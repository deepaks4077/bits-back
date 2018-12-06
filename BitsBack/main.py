"""
The main testing coding is launched from here
"""

import datasets
from bin_minst_experiment import original_bernoulli_example
from mnist_experiment import original_bbinomial_example

def main():
    """
    Run examples. Note: If an experiment fails in ans.pop, try adding more random bits
    """
    #original_bernoulli_example(100, 'bin_mnist_original_results.txt')
    original_bbinomial_example(100, 'mnist_original_results.txt')

    
def download_datasets():
    datasets.download_MNIST()

if __name__ == "__main__":
    main()