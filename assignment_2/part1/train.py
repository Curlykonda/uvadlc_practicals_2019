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

import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np

import torch
import torch.nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(
            seq_length=config.input_length,
            input_dim=config.input_dim,
            num_hidden=config.num_hidden,
            num_classes=config.num_classes,
            batch_size=config.batch_size,
            device=device
        )
    elif config.model_type == 'LSTM':
        model = LSTM(
            seq_length=config.input_length,
            input_dim=config.input_dim,
            num_hidden=config.num_hidden,
            num_classes=config.num_classes,
            batch_size=config.batch_size,
            device=device
        )

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=config.learning_rate)

    # evaluation metrics
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    results = []

    #best_acc = 0.0

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #transform into tensor representation
        s_inputs = batch_inputs.shape
        s_targets = batch_targets.shape
        #batch_inputs =batch_inputs.reshape((config.batch_size, config.input_dim))
        #inputs_torch = torch.tensor(batch_inputs, dtype=torch.float, device=device) #perhaps dtype=torch.int
        #targets_torch = torch.tensor(batch_targets, dtype=torch.float, device=device)  #

        #set gradients to zero
        optimizer.zero_grad()

        ############################################################################
        # QUESTION: what happens here and why?
        # Prevents exploding gradients by rescaling to a limit specified by config.max_norm
        # Forcing gradients to be within a certain norm to ensure reasonable updates
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        #forward pass
        predictions = model.forward(batch_inputs) #apply softmax?

        #compute loss
        loss = criterion(predictions, batch_targets)

        #backward pass & updates
        loss.backward()
        optimizer.step()

        accuracy = (predictions.argmax(dim=1) == batch_targets).float().mean().item() #

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.eval_freq == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            results.append([step, accuracy, loss])

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    res_path = Path.cwd()

    #if not res_path.exists():
    #    res_path.mkdir(parents=True)

    print("Saving results to {0}".format(res_path))

    res_path = res_path / 'results.csv'

    mode = 'a'
    if not res_path.exists():
        mode = 'w'

    col_names = ['step', 'acc', 'loss',
                 'Model', 'seq_length', 'input_dim', 'num_hidden',
                 'lr', 'train_steps', 'batch_size'] #'optimizer'

    with open(res_path, mode) as csv_file:
        if mode == 'w':
            csv_file.write('|'.join(col_names) + '\n')
        #for i in range(len(results)):

        steps, accs, losses = list(zip(*results))

        csv_file.write(
            f'{results[np.argmax(accs)][0]};{results[np.argmax(accs)][1]};{results[np.argmax(accs)][2]};'
            f'{config.model_type};{config.input_length};{config.input_dim};{config.num_hidden}'
            f'{config.learning_rate};{config.train_steps};{config.batch_size};' + '\n')

################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--eval_freq', type=int, default=100, help="Frequency of evaluating model")


    config = parser.parse_args()

    # Train the model
    train(config)