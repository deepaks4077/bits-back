"""
This file contains helper functions to pop and append given
a distribution and convert to start/frequency
"""
from ans import ANSCoder
from scipy.stats import norm

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
            symbols.append(pop(ans, start, freq))
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
            symbol = pop_symbol(start, frequency)
            symbols.append(symbol)

        return np.array(symbols)

    return pop

def create_bernoulli_buckets(p, precision):
    """
    Split the pdf according to the probability of the symbols (2 in this case)
    """
    buckets = np.array([ np.rint(    p    * (1 << precision) - 2) + 1,
                         np.rint(( 1 - p) * (1 << precision) - 2) + 1 ])
    
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
    data_shape = np.shape(probs)[:-1]
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

