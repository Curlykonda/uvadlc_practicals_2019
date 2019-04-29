# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        #params
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        #LSTM module
        self.LSTM = nn.LSTM(
            input_size=self.vocabulary_size,
            hidden_size=self.lstm_num_hidden,
            num_layers=self.lstm_num_layers
        )

        #linear module
        self.Linear = nn.Linear(
            in_features=self.lstm_num_hidden,
            out_features=self.vocabulary_size
        )

    def forward(self, x, hidden_states=None):
        '''


        :param x: input as torch.tensor, shape:
        :param hidden_states: tuple of torch.tensor, (last_hidden_state, last_cell_state)
        :return:
        '''

        #LSTM forward pass
        out, (hidden_s, cell_s) = self.LSTM(x, hidden_states)

        #Linear forward pass
        out = self.Linear(out)

        return out, (hidden_s, cell_s)
