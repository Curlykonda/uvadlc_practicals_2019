"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    """

    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}

    #Initialize weights using normal distribution
    # with mean = 0 and std = 0.0001
    self.params["weight"] = np.random.normal(loc=0.0,
                                             scale=0.0001,
                                             size= (out_features, in_features))
    #Initialize biases with 0.
    self.params["bias"] = np.zeros((out_features,1))

    #Initialize gradients with 0
    self.grads["weight"] = np.zeros((out_features, in_features))
    self.grads["bias"] = np.zeros((out_features, 1))

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module (shape: batch-size x features)
    Returns:
      out: output of the module

    """

    #whats the shape of input x? perhaps transpose
    out = np.dot(self.params["weight"], x.T) + self.params["bias"]

    #store intermediate variable for backward pass
    self.x = x

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module, x_tilde
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    #linear
    dx = np.dot(dout, self.params["weight"])

    #weights dL/dW
    self.grads["weight"] = np.dot(dout.T, self.x.T)

    #biases dL/db
    self.grads["bias"] = dout

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    #forward: map x-tilde -> x-activations := ReLU(x-tilde)
    out = np.maximum(x, 0)

    #store intermediate variable
    self.x = x

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    # dL/dx-tilde
    #np.diag(self.x > 0)
    dx = dout * (self.x > 0)

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    # forward: map x-tilde -> x-activations := Softmax(x-tilde)
    # Apply max trick for numerical stability
    numerator = np.exp(x - np.max(x,axis=1)) #sum column-wise
    denominator = numerator.sum(axis=1)

    out = numerator / denominator

    #store intermediate variable
    self.x = x
    self.out = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    #dx = dL/dx-tilde = dL/dx * dx/dx-tilde
    #dout = dL/dx  from the loss module
    out_sqrd = np.dot(self.out, self.out.T)

    #create tensor of shape batch-size x features x features with diagonal sub-matrices
    diag_out = np.zeros((self.out.shape[0], self.out.shape[1], self.out.shape[1]))
    diag_indices = np.arange(self.out.shape[1])
    diag_out[:, diag_indices, diag_indices] = self.out

    dxdx_tilde = diag_out - out_sqrd

    dx = np.einsum('ij, ijk -> ik', dout, dxdx_tilde)

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    """

    #compute cross entropy according to formula in assignment
    #out = np.sum(y * (-1) * np.log(x), axis=1).mean()
    #for mini batches compute mean of loss
    out = -1 * np.sum(y*np.log(x), axis=1)

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    #
    dx = -np.dot(y.T, np.diag(1/x))

    return dx
