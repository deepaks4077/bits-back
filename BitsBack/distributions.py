"""
This file contains helper functions to pop and append given
a distribution and convert to start/frequency
"""
from ans import ANSCoder
from scipy.stats import norm, beta, binom
from scipy.special import gammaln
import numpy as np
import torch
from mpmath import hyp3f2

def uniforms_append(precision):
    """
    Append symbols following the uniform distribution
    """
    def append(ans:ANSCoder, symbols):
        for symbol in reversed(symbols):
            start = symbol
            frequency = 1
            ans.append(start, frequency, precision)
        return
    return append

def uniforms_pop(precision, n):
    """
    Pop symbols following the uniform distribution
    """
    def pop(ans:ANSCoder):
        symbols = []
        for i in range(0, n):
            cf, pop = ans.pop(precision)
            start = cf
            freq = 1
            pop(ans, start, freq)
            symbols.append(cf)
        return symbols
    return pop

def distr_append(precision, cdfs):
    """
    Returns an append(ans, symbols) function that uses cdfs to 
    compute the start and frequency of symbols
    """
    def append(ans:ANSCoder, symbols):
        for symbol, cdf in reversed(list(zip(symbols, cdfs))):
            start = cdf(symbol)
            frequency = cdf(symbol + 1) - start
            ans.append(start, frequency, precision)
    return append

def distr_pop(precision, ppdfs, cdfs):
    """
    Returns a pop(ans), array given the prior probability and cdf
    """
    def pop(ans:ANSCoder):
        symbols = []

        for ppf, cdf in zip(ppdfs, cdfs):

            bits, pop_symbol = ans.pop(precision)
            index = ppf(bits)
            start = cdf(index)
            frequency = cdf(index + 1) - start
            pop_symbol(ans, start, frequency)
            symbols.append(index)

        return np.array(symbols)

    return pop

#
# Buckets, CDF, PPF and  append/pop for Bernoulli
#

def create_bernoulli_buckets(p, precision):
    """
    Split the pdf according to the probability of the symbols (2 in this case)
    """
    buckets = np.array([ np.rint(( 1 - p) * ((1 << precision) - 2)) + 1,
                         np.rint(    p    * ((1 << precision) - 2)) + 1 ])
    
    # fix the buckets if the sum is not correct
    bucket_sum = sum(buckets)
    if not bucket_sum == 1 << precision:
        i = np.argmax(buckets)
        buckets[i] += (1 << precision) - bucket_sum
    assert sum(buckets) == 1 << precision

    # return cumulative bucket sum (i.e like sum\integral on the pdf)
    return np.insert(np.cumsum(buckets), 0, 0)

def bernoulli_cdf(p, precision):
    """
    cdf of Bernoulli distribution given index
    """
    def cdf(s):
        cumulative_buckets = create_bernoulli_buckets(p, precision)
        return int(cumulative_buckets[s])
    return cdf

def bernoulli_ppf(p, precision):
    """
    Percent point function of Bernoulli distribution.
    The search sorted allows to find the right bucket for the given
    cumulative frequency (from rans)
    """
    def ppf(cf):
        cumulative_buckets = create_bernoulli_buckets(p, precision)
        return np.searchsorted(cumulative_buckets, cf, 'right') - 1
    return ppf

def bernoullis_append(probs, precision):
    """
    probs is a list of probabilities
    returns an append data following the bernoulli distribution
    """
    cdfs = [bernoulli_cdf(p, precision) for p in probs]
    def append(ans:ANSCoder, data):
        data = np.ravel(data)
        return distr_append(precision, cdfs)(ans, data)
    return append

def bernoullis_pop(probs, precision):
    """
    probs is a list of probabilities.
    Returns a pop function using the bernoulli distribution
    """
    data_shape = np.shape(probs) #[:-1]
    cdfs = [bernoulli_cdf(p, precision) for p in probs]
    ppfs = [bernoulli_ppf(p, precision) for p in probs]

    def pop(ans:ANSCoder):
        symbols = distr_pop(precision, ppfs, cdfs)(ans)
        return np.reshape(symbols, data_shape)

    return pop

def bernoulli_obs_append(precision):
    """
    Util function for VAE append bernoulli
    """
    def append_obs(probs):
        def append(ans:ANSCoder, data):
            return bernoullis_append(probs, precision)(ans, np.int64(data))
        return append
    return append_obs

def bernoulli_obs_pop(precision):
    """
    Util function for VAE pop bernoulli
    """
    def obs_pop(probs):
        def pop(ans:ANSCoder):
            data = bernoullis_pop(probs, precision)(ans)
            return torch.Tensor(data)
        return pop
    return obs_pop

#
# Buckets, CDF, PPF, append and pop for Beta-Binomial
#

def create_beta_binomial_buckets(n, a, b, precision):
    """
    Bad approximation of beta binomial buckets. A good solution would be to make a function that takes a target probability
    and computes the percent point function (ppf) using numerical methods. There is no closed form solution to the beta
    binomial ppf. It is approximate as a categorical distribution where each k is associated a probability
    """
    probs = []
    for k in range(0, n + 1):
        probs.append(beta_binomial_pdf_exact(k, n, a, b))
    probs = np.array(probs)
    probs = np.clip(probs, 1e-10, 1.0)
    buckets = [ np.rint( p * ((1 << precision) - 2)) + 1 for p in probs]

    bucket_sum = sum(buckets)
    if not bucket_sum == 1 << precision:
        i = np.argmax(buckets)
        buckets[i] += (1 << precision) - bucket_sum
    assert sum(buckets) == 1 << precision

    return np.insert(np.cumsum(buckets), 0, 0)

def beta_binomial_cdf_exact(k, n, a, b):
    """
    Use the hypergeometric function 3F2 to obtain the CDF of beta binomial
    """
    if k < 0:
        return 0.0
    elif k >= n:
        return 1.0
    else:
        pdf = beta_binomial_pdf(k, n, a, b)
        hg3F2 = float(nhyp3f2(1, -k, n-k+b, n-k-1, 1-k-a, 1))
        return pdf * hg3F2

def beta_binomial_log_pdf_exact(k, n, a, b):
    """
    Return the log of the pdf of the beta binomial distribution with parameters:
    k: input P(X=k)
    n: number of trials
    a: shape 1 > 0
    b: shape 2 > 0
    """
    assert a > 0
    assert b > 0
    assert k >= 0
    assert n >= 0

    numerator = gammaln(n+1) + gammaln(k+a) + gammaln(n - k + b) + gammaln(a+b)
    denominator = gammaln(k + 1) + gammaln(n - k + 1) + gammaln(n + a + b) + gammaln(a) + gammaln(b)
    logpdf = numerator - denominator
    return logpdf

def beta_binomial_pdf_exact(k, n, a, b):
    """
    Returns the result of the beta binomial pdf at k
    """
    return np.exp(beta_binomial_log_pdf_exact(k, n, a, b))

def beta_binomial_cdf(n, a, b, precision):
    """
    Compute CDF from buckets
    """
    def cdf(index):
        buckets = create_beta_binomial_buckets(n, a, b, precision)
        return int(buckets[index])
    return cdf

def beta_binomial_ppf(n, a, b, precision):
    """
    Approximate ppf given buckets
    """
    def ppf(cf):
        cumulative_buckets = create_beta_binomial_buckets(n, a, b, precision)
        return np.searchsorted(cumulative_buckets, cf, 'right') - 1
    return ppf

def beta_binomial_append(n_list, a_list, b_list, precision):
    """
    n, a, b are lists of parameters
    returns an append data following the bernoulli distribution
    """
    cdfs = [beta_binomial_cdf(n, a, b, precision) for (n, a, b) in zip(n_list, a_list, b_list)]
    def append(ans:ANSCoder, data):
        data = np.ravel(data)
        return distr_append(precision, cdfs)(ans, data)
    return append

def beta_binomial_pop(n_list, a_list, b_list, precision):
    """
    n_list, a_list, b_list : list of parameters
    Returns a pop function using the beta binomial distribution
    """
    data_shape = np.shape(n_list)
    cdfs = [beta_binomial_cdf(n, a, b, precision) for (n, a, b) in zip(n_list, a_list, b_list)]
    ppfs = [beta_binomial_ppf(n, a, b, precision) for (n, a, b) in zip(n_list, a_list, b_list)]

    def pop(ans:ANSCoder):
        symbols = distr_pop(precision, ppfs, cdfs)(ans)
        return np.reshape(symbols, data_shape)

    return pop

def beta_binomial_obs_append(precision):
    """
    Util function for VAE append beta binomial
    """
    def append_obs(n_list, a_list, b_list):
        def append(ans:ANSCoder, data):
            return beta_binomial_append(n_list, a_list, b_list, precision)(ans, np.int64(data))
        return append
    return append_obs

def beta_binomial_obs_pop(precision):
    """
    Util function for VAE pop beta binomial
    """
    def obs_pop(n_list, a_list, b_list):
        def pop(ans:ANSCoder):
            data = beta_binomial_pop(n_list, a_list, b_list, precision)(ans)
            return torch.Tensor(data)
        return pop
    return obs_pop




#
# Buckets, CDF and PPF for gaussians (used for the VAE)
#

def round_nearest_integer(array):
    return int(np.around(array))

def standard_gaussian_buckets(precision):
    buckets = np.float32(norm.ppf(np.arange(1 << precision + 1) / (1 << precision)))
    return buckets

def gaussian_latent_cdf(mean, stdd, prior_precision, posterior_precision):
    def cdf(index):
        x = standard_gaussian_buckets(prior_precision)[index]
        return round_nearest_integer(norm.cdf(x, loc=mean, scale=stdd) * (1 << posterior_precision))
    return cdf

def gaussian_latent_ppf(mean, stdd, prior_precision, posterior_precision):
    def ppf(cf):
        x = norm.ppf( (cf + 0.5) / (1 << posterior_precision), mean, stdd)
        return np.searchsorted(standard_gaussian_buckets(prior_precision), x, 'right') -1
    return ppf

def standard_gaussian_centers(precision):
    centers = np.float32(norm.ppf(( np.arange(1 << precision) + 0.5) / (1 << precision)))
    return centers

if __name__ == '__main__':

    buckets = create_beta_binomial_buckets(10, 600, 400, 8)
    print('Success')