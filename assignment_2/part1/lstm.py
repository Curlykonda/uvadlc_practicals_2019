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

import torch
import torch.nn as nn
import numpy as np

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu', mu=0, std=0.001):
        super(LSTM, self).__init__()

        # params
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        #mu = 0
        #std = 0.001

        #input modulation gate
        self.w_gx = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.w_gh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.b_g = nn.Parameter(torch.zeros(self.num_hidden).to(self.device))

        #input gate
        self.w_ix = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.w_ih = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.b_i = nn.Parameter(torch.zeros(self.num_hidden).to(self.device))

        #forget gate
        self.w_fx = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.w_fh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.b_f = nn.Parameter(torch.zeros(num_hidden).to(self.device))

        #output gate
        self.w_ox = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.w_oh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=mu, std=std).to(self.device))
        self.b_o = nn.Parameter(torch.zeros(num_hidden).to(self.device))

        #output
        self.w_ph = nn.Parameter(torch.Tensor(self.num_hidden, self.num_classes).normal_(mean=mu, std=std).to(self.device))
        self.b_p = nn.Parameter(torch.zeros(num_classes).to(self.device))

        #activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()



    def forward(self, x):

        #Initialise cell and hidden state
        h = nn.Parameter(torch.zeros(self.num_hidden).to(self.device))
        c = nn.Parameter(torch.zeros(self.num_hidden).to(self.device))

        #compute hidden states
        for idx in range(self.seq_length):
            x_i = x[:, idx].view(self.batch_size, 1).to(self.device)
            #input modulation
            g = self.tanh(x_i @ self.w_gx +
                          h @ self.w_gh +
                          self.b_g)
            #input gate
            i = self.sigmoid(x_i @ self.w_ix +
                          h @ self.w_ih +
                          self.b_i)

            #forget gate
            f = self.sigmoid(x_i @ self.w_fx +
                          h @ self.w_fh +
                          self.b_f)

            #output gate
            o = self.sigmoid(x_i @ self.w_ox +
                             h @ self.w_oh +
                             self.b_o)

            #cell state
            c = g * i + c * f #element-wise multiplication

            #hidden
            h = self.tanh(c) * o

        #compute output
        p = self.softmax(h @ self.w_ph + self.b_p)

        return p