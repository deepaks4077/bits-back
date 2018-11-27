"""
This class contains the code to encode / decode data with ANS
Based on https://github.com/rygorous/ryg_rans (public domain)
which implements a range asymmetric numeral system. 
The range part is from range coding which uses intervals to code the information.
The intervals are determined with a start value and it's frequency for the length of the interval.
See https://en.wikipedia.org/wiki/Range_encoding
"""

import numpy as np

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

    
    def __append(self, start, frequency):
        """
        Append a new symbol to the state using range [start, start + frequency)
        Assume that the sum of all frequencies is equal to 1 << precision
        """
        current_end = self.state[-1]
        max_value = ((self.l >> self.precision) << 32) * frequency

        # can't store anything in current_end state
        if( current_end >= max_value):
            #do something
            x = 0


    def __pop():
        """
        Remove symbol from the state
        """
        x = 0


if __name__ == "__main__":
    coder = AnsCoder(8)
    print(coder.state)