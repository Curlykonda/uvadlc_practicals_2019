"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    self.num_hidden = len(n_hidden)
    self.num_layers = self.num_hidden + 1

    self.lin_layer = []
    self.non_lin = []

    #initialize the layer 0 receiving the input
    self.lin_layer.append(LinearModule(n_inputs, n_hidden[0]))
    self.non_lin.append(ReLUModule())

    #initialize layers l = 1, ... , N-1
    for l in range(self.num_hidden - 1):
      self.lin_layer.append(LinearModule(n_hidden[i], n_hidden[i+1]))
      self.non_lin.append(ReLUModule())


    #initialize output layer N
    self.lin_layer.append(LinearModule(n_hidden[-1], n_classes))
    self.non_lin.append(SoftMaxModule())

    assert self.num_layers == len(self.lin_layer)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return
