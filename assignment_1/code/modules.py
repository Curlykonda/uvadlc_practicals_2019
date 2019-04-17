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
    shape_x = x.shape
    shape_w = self.params["weight"].shape # out_features x in_features

    #whats the shape of input x? perhaps transpose
    out = np.dot(self.params["weight"], x.T) + self.params["bias"]

    shape_out = out.shape # out_features x batch_size
    #store intermediate variable for backward pass
    self.x = x

    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module, x_tilde (shape: )
    Returns:
      dx: gradients with respect to the input of the module

    """

    #linear
    dx = np.dot(dout, self.params["weight"]) #shape: batch_size x in_features
    shape_dx = dx.shape

    #weights dL/dW
    shape_x = self.x.shape
    shape_dout = dout.shape
    self.grads["weight"] = np.dot(dout.T, self.x) #shape out_features x in_features

    #biases dL/db
    #self.grads["bias"] = dout.T
    #row-wise summation
    self.grads["bias"] = np.reshape(dout.sum(axis=0), self.grads["bias"].shape) #out_features x 1
    shape_bias = self.grads["bias"].shape

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module (shape: batch_size x in_features)
    Returns:
      out: output of the module

    
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
    """

    # dL/dx-tilde
    #np.diag(self.x > 0)
    shape_x = self.x.shape
    x_pos = (self.x > 0).astype(int)
    shape_dout = dout.shape
    #dx = np.dot(dout, np.diag(x))
    dx = dout * x_pos #shape: batch_size x out_features

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module (shape: batch-size x in_features
    Returns:
      out: output of the module

    """

    # forward: map x-tilde -> x-activations := Softmax(x-tilde)
    # Apply max trick for numerical stability, see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    x_max = x.max(axis=1).reshape((x.shape[0], 1))
    shape_x = x.shape
    shape_xmax = x_max.shape
    numerator = np.exp(x - x_max)
    denominator = np.reshape(numerator.sum(axis=1), (numerator.shape[0], 1)) #sum column-wise to sum up correspoding elements over multiple vectors
    shape_num = numerator.shape
    shape_den = denominator.shape
    out = numerator / denominator
    shape_out = out.shape
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
    """

    #dx = dL/dx-tilde = dL/dx * dx/dx-tilde
    #dout = dL/dx  from the loss module
    s_dout = dout.shape
    s_out = self.out.shape

    #out_sqrd = np.dot(self.out, self.out.T)
    #einsum convention simplifies matrix multiplication. instead of multiplying, summing and transposing we can use einsum, see http://ajcr.net/Basic-guide-to-einsum/
    #label axis of matrix A and B with ij and ik respectively
    #input: 2D arrays, output: 3D array with labels ijk
    out_sqrd = np.einsum('ij, ik->ijk', self.out, self.out) #shape: batch-size x features x features
    s_outsqrd = out_sqrd.shape

    #create tensor of shape batch-size x features x features with diagonal sub-matrices
    diag_out = np.zeros((self.out.shape[0], self.out.shape[1], self.out.shape[1]))
    diag_indices = np.arange(self.out.shape[1])
    diag_out[:, diag_indices, diag_indices] = self.out
    s_diag = diag_out.shape
    dxdx_tilde = diag_out - out_sqrd

    dx = np.einsum('ij, ijk -> ik', dout, dxdx_tilde) # shape: batch-size x features

    s_dx = dx.shape

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
