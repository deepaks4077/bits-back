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

class ANSCoder:

    def __init__(self):
        """
        Initialize an ANSCoder with the specified precision.
        State is the list of all 64 bit values required to keep track of all the information for ANS
        """
        
        self.state:[uint64] = []

        self.tail:uint64    = (1 << 32) - 1        # 0x 00 00 00 00 FF FF FF FF
        self.l:uint64       = (1 << 31)            # 0x 00 00 00 00 80 00 00 00

        self.state.append(self.l)

    def append(self, start, frequency, precision):
        """
        Append a new symbol to the state using range [start, start + frequency)
        Assume that the sum of all frequencies is equal to 1 << precision
        """
        x = self.state[0]
        max_value = ((self.l >> precision) << 32) * frequency
        # with 8 bit precision this is 0x 80 0000 0000 0000 * frequency.

        # can't store anything in current state head. Add a new value
        if( x >= max_value):
            #head is now the filled 32 bits
            self.state[0] = x & self.tail
            self.state.insert(0, x >> 32)
            x = self.state[0]
            

        # compute new value at the head
        # // : integer divison
        self.state[0] = ((x // frequency) << precision) + (x % frequency) + start
        
            
    def pop(self, precision):
        """
        Remove symbol from the state as bits, return a function
        to complete the pop operation given the start and requency
        of the returned bits.
        """
        x = self.state[0]
        mask = ((1 << precision) - 1)
        bits = x & mask         # cumulative frequency
        def pop(self, start, frequency):
            self.state[0] =  (frequency * (self.state[0] >> precision)) + bits - start
            
            # update state
            if self.state[0] < self.l:
                x = self.state.pop(0)
                self.state[0] = (x << 32) | self.state[0] 

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
        self.state = []
        head = array[0] << 32 | array[1]
        self.state.append(head)
        for i in range(2, len(array)):
            self.state.append(array[i])