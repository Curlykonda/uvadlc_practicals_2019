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


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        #params
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        mu = 0
        std = 0.01

        # input x: batch_size x input_dim
        #weight matrices
        self.w_hx = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden).normal_(mean=mu, std=std).to(self.device)) # num_hidden x input_dim
        self.w_hh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=mu, std=std).to(self.device)) # num_hidden x num_hidden
        self.w_ph = nn.Parameter(torch.Tensor(self.num_hidden, self.num_classes).normal_(mean=mu, std=std).to(self.device)) # num_classes x num_hidden

        #biases
        self.b_h = nn.Parameter(torch.zeros(num_hidden).to(self.device)) # num_hidden x 1
        self.b_p = nn.Parameter(torch.zeros(num_classes).to(self.device)) # num_classes x 1

        #hidden states
        #self.h_states = [nn.Parameter(torch.zeros(num_hidden))] * seq_length #or n_layers

        #output
        #self.p_t = nn.Parameter(torch.zeros(num_classes)) #

        self.tanh = nn.Tanh()


    def forward(self, x):
        #for seq_length
        # compute h_t  & p_t
        h = torch.zeros(self.num_hidden).to(self.device)

        for i in range(self.seq_length):
            '''
            if i == 0:
                prev_hidden = torch.zeros(self.num_hidden).to(self.device)
            else:
                prev_hidden = self.h_states[i-1]
            '''
            s_x = x.shape # batch_size x seq_length

            '''
            weight_matrix = [input_dim x output_dim]
            input = [batchsize x input_dim]
            bias = [output_dim]

            output = input @ weight + bias
            '''
            #x: batch_size x seq_len
            #w_hx: input_dim x hidden
            #w_hh: hidden x hidden
            #prev_h: hidden

            x_i = x[:, i].view(self.batch_size, 1)

            a = x_i @ self.w_hx + h @ self.w_hh + self.b_h
            h = self.tanh(a)
            s_h = h.shape

            p = h @ self.w_ph + self.b_p
            #self.h_states[i] = self.p_t

        return p