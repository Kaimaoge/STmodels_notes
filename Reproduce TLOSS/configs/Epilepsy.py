# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:40:19 2023

@author: AA
"""

class Config(object):
    def __init__(self):
        # model configs
        self.in_channels = 1
        self.channels = 8
        self.num_classes = 2
        self.out_channels = 128
        self.depth = 3

        self.kernel_size = 8
        self.reduced_size = 256
        
        self.batch_size = 128
        self.drop_last = True
        
        self.data_path = 'data/Epilepsy'
        self.device = 'cuda'
        
        self.nb_random_samples = 5
        self.negative_penalty = 1
        
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.num_epoch = 40