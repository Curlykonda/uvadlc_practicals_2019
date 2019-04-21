"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
from pathlib import Path

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  """
    # calculate accuracy over all predictions and average it
    # now we're dealing with tensors!
    accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()

    return accuracy


def train():
    """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## Prepare all functions
    lr = FLAGS.learning_rate
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    eval_freq = FLAGS.eval_freq
    data_dir = FLAGS.data_dir

    train_treshold = 1e-6  # if train loss below that threshold, training stops

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

    n_classes = y_test.shape[1]

    # reshape data to tensor representation
    #x_test = x_test.reshape((n_test_samples, n_inputs))
    x_test_torch = torch.tensor(x_test, dtype=torch.float, device=device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float, device=device)

    # initialize ConvNet model
    convnet_model = ConvNet(
        n_channels=x_dim[1],
        n_classes=n_classes
    ).to(device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(convnet_model.parameters(), lr=lr)

    # evaluation metrics
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    best_acc = 0.0
    #results = []

    # train the model
    print("Start training")
    for step in range(max_steps):

        # get mini-batch
        x_train, y_train = train_data.next_batch(batch_size)
        #x_train = x_train.reshape((batch_size, n_inputs))

        # transform to tensor representation
        x_train_torch = torch.tensor(x_train, dtype=torch.float, device=device)
        y_train_torch = torch.tensor(y_train, dtype=torch.float, device=device)  # labels for mb training set

        # set gradients to zero
        optimizer.zero_grad()

        # forward pass mb to get predictions as output
        out = convnet_model.forward(x_train_torch)

        # compute loss
        loss_mb = loss_fn.forward(out, y_train_torch.argmax(dim=1))

        # backward pass
        loss_mb.backward()
        optimizer.step()

        # evaluate training and validation set (pretty much the same as with Numpy)
        # perhaps modify learning rate?
        if (step % eval_freq == 0) or (step == max_steps - 1):
            print(f"Step: {step}")
            # compute and store training metrics
            loss_train.append(loss_mb.item())
            acc_train.append(accuracy(out, y_train_torch))
            print("TRAIN acc: {0:.4f}  & loss: {1:.4f}".format(acc_train[-1], loss_train[-1]))

            # compute and store test metrics
            # Note that we use the test set as validation set!! Only as an exception :P
            out_test = convnet_model.forward(x_test_torch)
            loss_val = loss_fn.forward(out_test, y_test_torch.argmax(dim=1))
            loss_test.append(loss_val.item())
            acc_test.append(accuracy(out_test, y_test_torch))
            print("TEST acc: {0:.4f}  & loss: {1:.4f}".format(acc_test[-1], loss_test[-1]))

            #results.append([step, acc_train[-1], loss_train[-1], acc_test[-1], loss_test[-1]])

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
    res_path = Path.cwd().parent / 'conv_pytorch_results'
    if not res_path.exists():
        res_path.mkdir(parents=True)
    print("Saving results to {0}".format(res_path))
    #Save an array to a binary file in NumPy .npy format.
    #np.save(res_path / 'loss_train', loss_train)
    #np.save(res_path / 'acc_train', acc_train)
    #np.save(res_path / 'loss_test', loss_test)
    #np.save(res_path / 'acc_test', acc_test)
    #Save array to csv file
    np.savetxt(res_path / 'loss_train.csv', loss_train, delimiter=',')
    np.savetxt(res_path / 'acc_train.csv', acc_train, delimiter=',')
    np.savetxt(res_path / 'loss_test.csv', loss_test, delimiter=',')
    np.savetxt(res_path / 'acc_test.csv', acc_test, delimiter=',')

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
    FLAGS, unparsed = parser.parse_known_args()

    main()
