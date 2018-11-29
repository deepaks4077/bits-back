"""
This class contains the code to encode / decode data with ANS
Based on https://github.com/rygorous/ryg_rans (public domain)
which implements a range asymmetric numeral system. 
The range part is from range coding which uses intervals to code the information.
The intervals are determined with a start value and it's frequency for the length of the interval.
See https://en.wikipedia.org/wiki/Range_encoding
"""

import numpy as np
from numpy.random import RandomState

uint64 = np.uint64
uint32 = np.uint32

class AnsCoder:

    def __init__(self, precision):
        """
        Initialize an AnsCoder with the specified precision.
        State is the list of all 64 bit values required to keep track of all the information for ANS
        """
        
        self.state:[uint64] = []

        self.tail:uint64    = ( 1 << 32) - 1        # 0x 00 00 00 00 FF FF FF FF
        self.l:uint64       = ( 1 << 31)            # 0x 00 00 00 00 80 00 00 00
        self.precision      = precision

        self.state.append(self.tail)

    
    def append(self, start, frequency):
        """
        Append a new symbol to the state using range [start, start + frequency)
        Assume that the sum of all frequencies is equal to 1 << precision
        """
        x = self.state[0]
        max_value = ((self.l >> self.precision) << 32) * frequency
        # with 8 bit precision this is 0x 80 0000 0000 0000 * frequency.

        # can't store anything in current state head. Add a new value
        if( x >= max_value):
            #head is now the filled 32 bits
            self.state[0] = x & self.tail
            self.state.insert(0, x >> 32)
            x = self.state[0]
            

        # compute new value at the head
        # // : integer divison
        self.state[0] = ((x // frequency) << self.precision) + (x % frequency) + start
        
            
    def pop(self):
        """
        Remove symbol from the state as bits, return a function
        to complete the pop operation given the start and requency
        of the returned bits.
        """
        x = self.state[0]
        mask = ((1 << self.precision) - 1)
        bits = x & mask
        def pop(self, start, frequency):
            self.state[0] =  frequency * (x >> self.precision) + cf - start
            x = self.state[0]
            # update state
            if x < rans_l:
                x = self.state.pop(0)
                x = (x << 32) | self.state[0] 
                self.state[0] = x

        return bits, pop
        
    def to_array(self):
        """
        Output a numpy array of uint32s containing the state
        """
        result:[uint32] = []
        # the head of the state is the only one which is different
        x = self.state[0]
        result.append(x >> 32)
        result.append(x)

        for i in range(1, len(self.state)):
            x = self.state[i]
            result.append(x)
        
        result = np.array(result).astype(uint32)
        return result

    def from_array(self, array):
        """
        Set the current state to the given array.
        Assume array has at least 2 elements and is of type uint32
        """
        self.state = [self.tail]
        head = array[0] << 32 | array[1]
        self.state.append(head)
        for i in range(2, len(array)):
            self.state.append(array[i])



def test_ans(length):
    """
    Test and example on how to use the encoder/decoder
    """
    
    rng = RandomState(0)

    #Generate random starts
    starts = rng.randint(0, 256, size=length)

    #Generate random freqs with correct size
    freqs = rng.randint(1, 256, size=length) % (256 - starts)
    for i in range(0, length):
        freq = freqs[i]
        if freq == 0:
            freqs[i] = 1
    # verify that start + freq is always <= 1<<precision
    assert np.all(starts + freqs <= 256)

    # Source Coding Theorem lower bound on compression

    minimal_number_bits = np.ceil(np.sum(np.log2(256 / freqs)))
    print("Minimal number of bits required to store " + str(length) + " symbols: " + str(int(minimal_number_bits)) + " bits\n" )

    # We now encode the symbols using the range ANS coder
    ans = AnsCoder(8)

    for i in range(0, length):
        start = starts[i]
        frequency = freqs[i]
        ans.append(start, frequency)
        
    compressed_array = ans.to_array()
    assert compressed_array.dtype == uint32
    compressed_size = len(compressed_array) * 32
    print("Compressed symbols size: " + str(compressed_size) + " bits")

    # We now decode the symbols and verify the results



if __name__ == "__main__":
    test_ans(1000)