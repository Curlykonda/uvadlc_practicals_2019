"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
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

    super(MLP, self).__init__() #inheritance

    #check n_hidden
    if len(n_hidden) > 0:
        #add first layer that maps input features to dimension of hidden
        #self.modules.append(nn.Linear(n_inputs, n_hidden[0]))
        #self.modules.append(nn.ReLU)
        self.modules = [
            nn.Linear(n_inputs, n_hidden[0]),
            nn.ReLU()
        ]

        #add modlues for hidden layers comprising a linear module followed by non-linear
        #map input dimension to output dimension specified in list n_hidden
        for i in range(len(n_hidden) -1):
            #self.modules.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            #self.modules.append(nn.ReLU)
            self.modules += [
                nn.Linear(n_hidden[i], n_hidden[i+1]),
                nn.ReLU()
            ]

        #add last layer that maps to classes
        #self.modules.append(nn.Linear(n_hidden[-1], n_classes))
        #self.modules.append(nn.Softmax())
        self.modules += [
            nn.Linear(n_hidden[-1], n_classes),
            #nn.Softmax()
        ]
    else:
        #multinomial logistic regression
        self.modules = [nn.Linear(n_inputs, n_classes)]

    #nn.ModuleList = self.modules

    #nn.ModuleList = self.modules
    self.model = torch.nn.Sequential(*self.modules)
    '''
    These belong in the training file!
    model.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    '''


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

    out = self.model(x)

    return out
