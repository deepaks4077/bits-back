"""
The main testing coding is launched from here
"""

import datasets
from bin_minst_experiment import    original_bernoulli_example, bernoulli_100_100_40_20_False_5, bernoulli_100_100_20_20_False_3
from bin_minst_experiment import    bernoulli_100_150_20_20_False_5, bernoulli_100_150_40_20_False_5

from mnist_experiment import original_bbinomial_example


def main():
    """
    Run examples. Note: If an experiment fails in ans.pop, try adding more random bits
    """

    #original_bernoulli_example(200, 'bin_mnist_original_results.txt')

    #original_bbinomial_example(200, 'mnist_original_results.txt')

    # Batch size, hidden size, latent size, epoch, CUDA, e^-learning_rate

    bernoulli_100_100_40_20_False_5(200, 'bin_mnist_100_100_40_20_False_5_results.txt')
    bernoulli_100_100_20_20_False_3(200, 'bin_mnist_100_100_20_20_False_3_results.txt')
    bernoulli_100_150_20_20_False_5(200, 'bin_mnist_100_150_20_20_False_5_results.txt')
    bernoulli_100_150_40_20_False_5(200, 'bin_mnist_100_150_40_20_False_5_results.txt')
    

    
def download_datasets():
    datasets.download_MNIST()

if __name__ == "__main__":
    main()