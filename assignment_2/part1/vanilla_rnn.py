################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import dataset.py

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # input x: batch_size x input_dim
        #weight matrices
        self.w_hx = nn.Parameter(torch.ones(num_hidden, input_dim)) # num_hidden x input_dim
        self.w_hh = nn.Parameter(torch.ones(num_hidden, num_hidden)) # num_hidden x num_hidden
        self.w_ph = nn.Parameter(torch.ones(num_classes, num_hidden)) # num_classes x num_hidden

        #biases
        self.b_h = nn.Parameter(torch.zeros(num_hidden, 1)) # num_hidden x 1
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1)) # num_classes x 1

        #hidden states
        self.h_states = [nn.Parameter(torch.zeros(num_hidden, batch_size))] * seq_length #or n_layers

        #output
        self.p_t = nn.Parameter(torch.zeros(num_classes, batch_size)) #


    def forward(self, x):
        #for seq_length
        # compute h_t  & p_t
        for i in range(len(self.h_states)):
            self.h_states[i] = nn.Tanh(torch.sum(torch.mul(self.w_hx, x) + torch.mul(self.w_hh, self.h_states[i-1]), self.b_h))
            self.p_t = torch.sum(torch.mul(self.w_ph, self.h_states[i]), self.b_p)
