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

import os
import time
from datetime import datetime
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import sys
sys.path.append("..")

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

################################################################################

torch.manual_seed(43)
torch.cuda.manual_seed(43)
np.random.seed(43)


def create_checkpoint(check_dir, filename, dataset, model, optimizer, results, step):

    checkpoint_path = check_dir / (f'{filename}_step_{step}.pt')

    print(f"Saving checkpoint to {checkpoint_path}")

    torch.save(
        {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'dataset': dataset
        },
        checkpoint_path
    )

    print("Saved.")

def load_checkpoint(checkpoint_path, model, optimizer):
    #load checkpoint saved at checkpoint_path

    checkpoint = torch.load(checkpoint_path)
    dataset = checkpoint['dataset']
    step = checkpoint['step'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    results = checkpoint['results']

    return dataset, results, step

def char2onehot(batch, vocab_size, device='cpu'):
    '''
    Encodes the characters of the batch to one-hot vectors based on the vocab size
    '''

    #create tensor of zeros of shape batch_size x vocab_size
    one_hot_dim = [*batch.shape, vocab_size]
    one_hots = torch.zeros(one_hot_dim, device=device)
    '''
    torch.scatter_(dim, index, src)
    src = 1: these are the values that will be placed at the positions from batch.unsqueeze
    dim = 2 to create one-hot encoding along the third axis to get batch_size x seq_length x n
    
    '''
    one_hots.scatter_(2, batch.unsqueeze(dim=2), 1)

    return one_hots

def compute_acc(predictions, true_labels):
    acc = (predictions.argmax(dim=2) == true_labels).sum().float()
    acc /= (predictions.shape[0] * predictions.shape[1])

    return acc

def sample_from_model(predictions, temperature):

    sampled_char = ''


    '''
    create distribution over softmax probabilites / temperature
    sample integer from that distribution 
    '''

    s_pred = predictions.shape

    #softmax before or after balancing?
    output_dist = predictions.view(-1).div(temperature).exp()
    balanced_pred = (predictions.squeeze() / temperature).softmax(dim=0)

    sampled_char = torch.multinomial(balanced_pred, 1)[0]
    sampled_char = balanced_pred.multinomial(dim=1)

    return sampled_char


def generate_text(start_char, sentence_length, model, dataset, temperature=1.0, device='cpu'):

    #add start char to text snippet
    text_example = [start_char]

    #convert start_char to torch
    start_char = torch.tensor([start_char], dtype=torch.long).to(device)

    #s_start = start_char.shape
    #text_example = start_char.view(-1).tolist()
    #one_hot_char = char2onehot(start_char, dataset.vocab_size, device)

    #to avoid computing gradients for parameters we use torch.no_grad() for sampling
    with torch.no_grad():

        #convert to one-hot for forward pass
        start_char_oh = char2onehot(start_char, dataset.vocab_size, device)

        # initialise hidden and cell state
        pred, (hidden, cell) = model(start_char_oh) #returns tensor?

        s_pred = pred.shape

        #greedy sampling
        sampled_char = pred.argmax(dim=2)[-1, :].item()

        #balanced sampling
        sampled_char = sample_from_model(pred, temperature)

        #sampled_char = sample_from_model(pred, temperature)
        text_example.append(sampled_char)

        while len(text_example) < sentence_length:
            #sample from model
            sampled_char_oh = char2onehot(sampled_char, dataset.vocab_size, device)

            #forward pass to generate predictions, hidden and cell states
            pred, (hidden, cell) = model.forward(sampled_char_oh, (hidden, cell))

            #sample from predictions
            sampled_char = pred.argmax(dim=2)[-1, :].item()

            #add char to text example
            text_example.append(sampled_char)

        #map the char ids to a string
        text_example = dataset.convert_to_string(text_example)

    return text_example



def train(config):

    #Initialize params
    #temperatures = [0.5, 1.0, 2.0]
    sampled_texts = {t:[] for t in config.temperatures}

    # Initialize the device which to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(
        filename=config.txt_file,
        seq_length=config.seq_length
    )
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        device=device
    ).to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config.learning_rate
    )

    #Initialise metrics and results dir
    results = []
    text_samples = []

    # make the results directory (if it doesn't exist)
    res_path = Path.cwd() / 'output'
    res_path.mkdir(parents=True, exist_ok=True)

    check_path = Path.cwd() / config.check_path
    check_path.mkdir(parents=True, exist_ok=True)

    print_setting(config)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #convert input to one-hot encoding and tensor representation
        x_torch = torch.stack(batch_inputs, dim=1).to(device) #shape: batch_size x seq_length
        x_torch = char2onehot(x_torch, dataset.vocab_size)

        y_torch = torch.stack(batch_targets, dim=1).to(device)

        optimizer.zero_grad()

        #forward pass of mini-batch
        predictions, _ = model(x_torch)

        #avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        #y_torch: batch_size x seq_len
        #pred: batch_size x seq_len x vocab_size
        #transpose dimensions to compute loss
        loss = criterion(predictions.transpose(2,1), y_torch) #perhaps transpose predictions?

        #accuracy = torch.sum(predictions.argmax(dim=1) == y_torch).float().mean().item()
        accuracy = compute_acc(predictions, y_torch)

        #save stats
        results.append([step, accuracy, loss.float().item()])



        #backward propagation
        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss.item()
            ))


            results.append([step, accuracy, loss.item()])

        if step == config.sample_every:
            # Generate some sentences by sampling from the model

            #generate random starting char
            start_char = np.random.randint(0, high=dataset.vocab_size)

            for t in config.temperatures:
                sampled_text = generate_text(start_char, config.seq_length, model, dataset,
                                             t, device=device)


                text_samples[t].append([step, sampled_text])
                print(f"Sampled at step {step} with temp {t}: {sampled_text}")

            #create checkpoint
            create_checkpoint(check_path, filename='grimm', dataset=dataset,
                              model=model, optimizer=optimizer, results=results, step=step)


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655

            # save model
            create_checkpoint(check_path, filename='grimm', dataset=dataset,
                              model=model, optimizer=optimizer, results=results, step=step)
            break

    print('Done training.')

    #save results





def print_setting(config):
    for key, value in vars(config).items():
        print("{0}: {1}".format(key, value))

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, default='./assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')
    parser.add_argument('--temperatures', type=float, default=[0.5, 1.0, 2.0], help='Parameter to balance greedy and random strategy for sampling')

    # Misc params
    parser.add_argument('--check_path', type=str, default="checkpoints", help='Output path for checkpoints')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
