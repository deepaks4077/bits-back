"""
This file contains the functions to output the result of a encoding/decoding experiment
"""
import numpy as np

class Result():

    def __init__(self):
        """
        Initialize variables
        """
        self.image_count:int    = 0
        self.method_name:str    = ''
        self.exp_name:str       = ''
        self.seed:int           = 0
        self.path_to_model:str  = ''
        self.image_shape        = None
        self.encode_success     = False
        self.decode_success     = False 
        self.random_bits_count  = 0
        self.original_length    = 0
        self.compressed_length  = 0
        self.prior_precision    = 0
        self.latent_precision   = 0
        self.posterior_precision= 0
        self.latent_size        = 0
        self.hidden_size        = 0

    def to_file(self, path:str):
        """
        Store results to file
        """
        with open(path, "w") as file:

            def __writeline(string=''):
                file.write( string + '\n')

            __writeline('Results for experiment ' + self.exp_name)
            __writeline('Method: ' + self.method_name)
            __writeline()
            __writeline('Experimental setup:')
            __writeline('RandomState seed: ' + str(self.seed))
            __writeline('Number of images: '+ str(self.image_count))
            __writeline('Image shape: ' + str(self.image_shape))
            __writeline('Model path: ' + self.path_to_model)
            __writeline('Encode/Decode successful: ' +str(self.encode_success | self.decode_success))
            __writeline('Number of random bits: ' + str(self.random_bits_count))
            __writeline('Prior precision: ' + str(self.prior_precision))
            __writeline('Latent precision: ' + str(self.latent_precision))
            __writeline('Posterior precision: ' + str(self.posterior_precision))
            __writeline('Latent dimensions: ' + str(self.latent_size) )
            __writeline('Hidden dimensions: ' + str(self.hidden_size))
            __writeline()
            __writeline('Results:')
            __writeline('Original size: ' + str(self.original_length) + ' bits')
            __writeline('Encoded size: '  + str(self.compressed_length) + ' bits')

            bits_per_pixel = float(self.compressed_length - self.random_bits_count) / (self.image_count * self.image_shape[0] * self.image_shape[1])

            __writeline('Bits per pixel: ' + str(bits_per_pixel))

            


    
