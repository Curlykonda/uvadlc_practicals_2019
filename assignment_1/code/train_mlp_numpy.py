"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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

    # for each input vector in batch
    # get softmax probs through forward pass
    # compare to true label
    # compute accuracy by counting correctly classified

    # predictions.argmax(axis=1) == targets.argmax(axis=1)
    accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()

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

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # set training parameters
    lr = FLAGS.learning_rate
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    eval_freq = FLAGS.eval_freq #Frequency of evaluation on the test set
    data_dir = FLAGS.data_dir
    plot_results = FLAGS.plot
    train_treshold = 1e-6 #if train loss below that threshold, training stops

    # evaluation metrics
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []

    #load input data
    cifar10 = cifar10_utils.get_cifar10(data_dir, one_hot=True)

    #get test data
    x_test = cifar10["test"].images
    y_test = cifar10["test"].labels

    #determine dimension of data
    x_dim = x_test.shape
    n_test_samples = x_dim[0] #number of test samples
    #images of size 32 x 32 x 3
    n_inputs = x_dim[1] * x_dim[2] * x_dim[3] #channels * height * width
    #reshape test images to fit MLP input
    x_test = x_test.reshape((n_test_samples, n_inputs))

    n_classes = y_test.shape[1]

    #initialize MLP
    my_MLP = MLP(n_inputs = n_inputs,
                 n_hidden= dnn_hidden_units,
                 n_classes= n_classes)

    #define loss function
    loss_fn = CrossEntropyModule()

    print("Start training")

    for step in range(max_steps):
        #get new minibatch
        x_train, y_train = cifar10["train"].next_batch(batch_size)

        # reshape x that each sample is represented by a vector
        x_train = x_train.reshape((batch_size, n_inputs))

        #forward pass of minibatch
        out = my_MLP.forward(x_train) #here output are softmax probabilities

        #compute loss
        loss_mb = loss_fn.forward(out, y_train)
        #compute gradient of loss
        grad_loss = loss_fn.backward(out, y_train)

        #perform backpropagation
        my_MLP.backward(grad_loss) #backprop loss from layer to layer where gradients are save in each module

        #update parameters with SGD
        for l in my_MLP.lin_layer:
            l.params["weight"] = l.params["weight"] - lr*l.grads["weight"]
            l.params["bias"] = l.params["bias"] - lr * l.grads["bias"]

        #perhaps modify learning rate?
        if (step%eval_freq == 0) or (step == max_steps-1):
            print(f"Step: {step}")
            #compute and store training metrics
            loss_train.append(loss_mb)
            acc_train.append(accuracy(out, y_train))
            print("TRAIN acc: {0:.4f}  & loss: {1:.4f}".format(acc_train[-1], loss_train[-1]))

            #compute and store test metrics
            out_test = my_MLP.forward(x_test)
            loss_test.append(loss_fn.forward(out_test, y_test))
            acc_test.append(accuracy(out_test, y_test))
            print("TEST acc: {0:.4f}  & loss: {1:.4f}".format(acc_test[-1], loss_test[-1]))

        #Early stop when training loss below threshold?

            if len(loss_train) > 20:
                prev_losses = loss_train[-20:-10]
                cur_losses = loss_train[-10:]
                if (prev_losses - cur_losses) < train_treshold:
                    print("Training stopped early at step {0}".format(step+1))
                    break

    print("Finished training")
    res_path = Path.cwd().parent / 'mlp_np_results'
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

    if plot_results:
        #plot results
        plt.plot(loss_train, label="Training")
        plt.plot(loss_test, label="Test")
        plt.xlabel('Time steps')
        plt.ylabel('Loss')
        plt.title("Loss of minibatches over time steps")
        plt.legend()
        plt.show()

        plt.plot(acc_train, label="Training")
        plt.plot(acc_test, label="Test")
        plt.xlabel('Time steps')
        plt.ylabel('Accuracy')
        plt.title("Accuracy of minibatches over time steps")
        plt.legend()
        plt.show()


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
    parser.add_argument('--plot', type=bool, default=False,
                        help='Plot loss and accuracy curves')
    FLAGS, unparsed = parser.parse_known_args()

    main()
