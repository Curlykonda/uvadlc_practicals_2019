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

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

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

    ########################
    # PUT YOUR CODE HERE  #
    #######################
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
    my_MLP = MLP(
        n_inputs = n_inputs,
        n_hidden= dnn_hidden_units,
        n_classes= n_classes
    ).to(device)

    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(my_MLP.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(my_MLP.parameters(), lr=lr)

    optimizer.zero_grad()

    #define loss function
    loss_fn = nn.CrossEntropyLoss()

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
  Main function
  """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

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

    FLAGS, unparsed = parser.parse_known_args()

    main()
