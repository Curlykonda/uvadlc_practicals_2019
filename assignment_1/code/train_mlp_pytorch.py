"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_pytorch import MLP
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OPTIMIZER_DEFAULT = 'SGD'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    targets: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  """
    #calculate accuracy over all predictions and average it
    # now we're dealing with tensors!
    accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()

    return accuracy


def train():
    """
  Performs training and evaluation of MLP model.
    """
    print_flags()
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    lr = FLAGS.learning_rate
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    eval_freq = FLAGS.eval_freq
    data_dir = FLAGS.data_dir
    optim_type = FLAGS.optimizer
    #plot_results = FLAGS.plot
    train_treshold = 1e-6  # if train loss below that threshold, training stops

    # evaluation metrics
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []

    # load input data
    cifar10 = cifar10_utils.get_cifar10(data_dir, one_hot=True)

    # get test data
    x_test = cifar10["test"].images
    y_test = cifar10["test"].labels
    train_data = cifar10["train"]

    # determine dimension of data
    x_dim = x_test.shape
    n_test_samples = x_dim[0]  # number of test samples
    # images of size 32 x 32 x 3
    n_inputs = x_dim[1] * x_dim[2] * x_dim[3]  # channels * height * width
    # reshape test images to fit MLP input
    x_test = x_test.reshape((n_test_samples, n_inputs))

    n_classes = y_test.shape[1]

    #reshape data to tensor representation
    x_test = x_test.reshape((n_test_samples, n_inputs))
    x_test_torch = torch.tensor(x_test, dtype=torch.float, device=device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float, device=device)

    #initialize MLP model
    mlp_model = MLP(
        n_inputs = n_inputs,
        n_hidden= dnn_hidden_units,
        n_classes= n_classes
    ).to(device)

    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(mlp_model.parameters(), lr=lr)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)
    elif optim_type == 'Adadelta':
        optimizer = torch.optim.Adadelta(mlp_model.parameters(), lr=lr)


    optimizer.zero_grad()

    #define loss function
    loss_fn = nn.CrossEntropyLoss()

    # evaluation metrics
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    best_acc = 0.0
    results = []

    #train the model
    print("Start training")
    for step in range(max_steps):

        #get mini-batch
        x_train, y_train = train_data.next_batch(batch_size)
        x_train = x_train.reshape((batch_size, n_inputs))

        #transform to tensor representation
        x_train_torch = torch.tensor(x_train, dtype=torch.float, device=device)
        y_train_torch = torch.tensor(y_train, dtype=torch.float, device=device) #labels for mb training set

        #set gradients to zero
        optimizer.zero_grad()

        #forward pass mb to get predictions as output
        out = mlp_model.forward(x_train_torch)

        #compute loss
        loss_mb = loss_fn.forward(out, y_train_torch.argmax(dim=1))

        #backward pass
        loss_mb.backward()
        optimizer.step()

        #evaluate training and validation set (pretty much the same as with Numpy)
        # perhaps modify learning rate?
        if (step % eval_freq == 0) or (step == max_steps - 1):
            print(f"Step: {step}")
            # compute and store training metrics
            loss_train.append(loss_mb.item())
            acc_train.append(accuracy(out, y_train_torch))
            print("TRAIN acc: {0:.4f}  & loss: {1:.4f}".format(acc_train[-1], loss_train[-1]))

            # compute and store test metrics
            # Note that we use the test set as validation set!! Only as an exception :P
            out_test = mlp_model.forward(x_test_torch)
            loss_val = loss_fn.forward(out_test, y_test_torch.argmax(dim=1))
            loss_test.append(loss_val.item())
            acc_test.append(accuracy(out_test, y_test_torch))
            print("TEST acc: {0:.4f}  & loss: {1:.4f}".format(acc_test[-1], loss_test[-1]))

            results.append([step, acc_train[-1], loss_train[-1], acc_test[-1], loss_test[-1]])

            if acc_test[-1] > best_acc:
                best_acc = acc_test[-1]
                print("New BEST acc: {0:.4f}".format(best_acc))

            # Early stop when training loss below threshold?
            if len(loss_train) > 20:
                prev_losses = loss_test[-2]
                cur_losses = loss_test[-1]
                if (prev_losses - cur_losses) < train_treshold:
                    print("Training stopped early at step {0}".format(step + 1))
                    break
    print("Finished training")
    print("BEST acc: {0:.4f}".format(best_acc))


    res_path = Path.cwd().parent / 'mlp_pytorch_results'

    if not res_path.exists():
        res_path.mkdir(parents=True)

    print("Saving results to {0}".format(res_path))

    #model_path.mkdir(parents=True, exist_ok=True)
    #model_path = model_path / 'mlp_pytorch.csv'
    res_path = res_path / 'mlp_pytorch.csv'

    mode = 'a'
    if not res_path.exists():
        mode = 'w'

    col_names =  ['step', 'train_acc', 'train_loss', 'test_acc', 'test_loss',
                  'lr', 'max_steps', 'batch_size', 'dnn_hidden_units', 'optimizer']

    with open(res_path, mode) as csv_file:
        if mode == 'w':
            csv_file.write('|'.join(col_names) + '\n')
        for i in range(len(results)):
            csv_file.write(
                f'{results[i][0]};{results[i][1]};{results[i][2]};{results[i][3]};{results[i][4]}'
                f'{lr};{max_steps};{batch_size};{dnn_hidden_units};{optim_type};' + '\n')

            #results.append([step, acc_train[-1], loss_train[-1], acc_test[-1], loss_test[-1]])
    return results
    #plot results

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def auto_optimize():

    #define different net params

    learning_rates = np.logspace(-4.0, -3.0, num=2, base=10)
    learning_rates = np.append(learning_rates, learning_rates*5).tolist()
    max_steps = range(500, 4000, 500)
    batch_sizes = range(20, 400, 40) #no offense but it's not something like 64, 128, ...
    optimizers = ['SGD', 'Adam', 'Adadelta']

    dnn_hidden_possibilities = [
        '1000', '2000', #shallow
        '100, 100', '500,500',
        '100, 100, 100', '500,500,500,500',
        '500, 400, 300, 100', '1000,10,10,10,10',
        '100, 100, 100, 100, 100', '500,500,500,500,500', #deep
        '800,400,200,100,50', '1000,500,300,100,50'
    ]

    #determine params setting
    net_settings = []
    i = 0
    #while len(net_settings) < 10:
    while i < len(dnn_hidden_possibilities):
        setting = {}
        #setting["lr"] = random.choice(learning_rates)
        #setting["lr"] = learning_rates[i]
        setting["lr"] = 1e-3
        setting["steps"] = random.choice(max_steps)
        setting["batch"] = random.choice(batch_sizes)
        setting["opt"] = random.choice(optimizers)
        #setting["nhidden"] = random.choice(dnn_hidden_possibilities)
        setting["nhidden"] = dnn_hidden_possibilities[i]
        net_settings.append(setting)
        i += 1

    best_acc = 0.0
    best_setting = []

    for setting in net_settings:
        FLAGS.learning_rate = setting["lr"]
        #FLAGS.max_steps = setting["steps"]
        #FLAGS.batch_size = setting["batch"]
        #FLAGS.optimizer = setting["opt"]
        FLAGS.dnn_hidden_units = setting["nhidden"]

        # run training
        result = train()[-1]

        # evluate results + save best model
        if result[3] > best_acc:
            best_acc = result[3]
            print("New BEST acc: {0:.4f}".format(best_acc))

def main():
    """
  Main function
  """
    # Print all Flags to confirm parameter settings


    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    if FLAGS.auto_opt:
        auto_optimize()
    else:
        # Run the training operation
        train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        help='Type of optimizer: SGD, Adam, Adadelta')

    parser.add_argument('--auto_opt', type=bool, default=True,
                        help='Set to True to train model with different settings')

    FLAGS, unparsed = parser.parse_known_args()

    main()
