"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """


    #initialize
    super(ConvNet, self).__init__() #inherit cool properties from parent class

    #build VGG architecture

    #create blocks
    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels=n_channels, out_channels=64, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    )

    self.block2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    )

    self.block3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    )

    self.block4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    )

    self.block5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    )

    self.block6 = nn.Sequential(
        nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0)
    )

    self.block7 = nn.Sequential(
        nn.Linear(in_features=512, out_features=n_classes)
    )

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """

    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out = self.block4(out)
    out = self.block5(out)
    out = self.block6(out)
    s_out = out.shape #shape: batch_size x features x 1 x 1
    #out = self.block7(out)
    out = self.block7(out.squeeze()) #Returns a tensor with all the dimensions of input of size 1 removed.

    return out
