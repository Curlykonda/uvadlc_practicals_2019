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
import itertools
import sys
sys.path.append("..")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        #forward pass
        predictions = model.forward(batch_inputs) #apply softmax?

        #compute loss
        loss = criterion(predictions, batch_targets)

        #backward pass & updates
        loss.backward()
        optimizer.step()

        accuracy = (predictions.argmax(dim=1) == batch_targets).float().mean().item()

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

            l = loss.float().item()
            results.append([step, accuracy, loss.float().item()])

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            save_results(results)
            break

    print('Done training.')

    #return results

def save_results(results):

    N = config.n_avg

    res_path = Path.cwd() / 'output'


    if not res_path.exists():
        res_path.mkdir(parents=True, exist_ok=True)

    print("Saving results to {0}".format(res_path))

    res_path = res_path / 'results.csv'

    mode = 'a'
    if not res_path.exists():
        mode = 'w'

    col_names = ['mean acc', 'std acc', 'mean loss', 'std loss',
                 'Model', 'seq_length', 'input_dim', 'num_hidden',
                 'lr', 'train_steps', 'batch_size'] #'optimizer'

    #, encoding="latin-1", errors="surrogateescape"
    with open(res_path, mode) as csv_file:
        if mode == 'w':
            csv_file.write('|'.join(col_names) + '\n')
        #for i in range(len(results)):

        steps, accs, losses = list(zip(*results))

        #compute average over last N iteration
        mean_acc = np.mean(accs[-N:])
        mean_loss = np.mean(losses[-N:])
        std_acc = np.std(accs[-N:])
        std_loss = np.std(losses[-N:])

        csv_file.write(
            f'{mean_acc};{std_acc};{mean_loss};{std_loss};'
            f'{config.model_type};{config.input_length};{config.input_dim};{config.num_hidden};'
            f'{config.learning_rate};{config.train_steps};{config.batch_size}' + '\n')

################################################################################
 ################################################################################

def experiment(config):

    num_iterations = 3

    lrs = [1e-3, 1e-4]
    seq_lengths = range(5, 105, 5)

    settings = list(itertools.product(*[lrs, seq_lengths]))

    for setting in settings:
        # set configurations
        config.learning_rate = setting[0]
        config.input_length = setting[1]
        # config.num_hidden = num_hidden
        # config.batch_size = batch_size

        #results = []
        #train for multiple initialisations because of stochasticity
        #for i in range(num_iterations):
        train_res = train(config)

        #    steps, accs, losses = list(zip(*train_res))

        #   results.append([i, np.mean(accs[-config.n_avg:]), np.mean(losses[-config.n_avg:])])

        #save_results(train_res)

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
    parser.add_argument('--n_avg', type=int, default=3, help="Averages results over last N batches")
    parser.add_argument('--experiment', type=bool, default=False, help="Run experiments")


    config = parser.parse_args()

    # Train the model
    if config.experiment:
        experiment(config)
    else:
        train(config)

