import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    """
    super(CustomBatchNormAutograd, self).__init__()

    #save variables in object
    self.epsilon = eps
    self.n_neurons = n_neurons

    self.gamma = nn.Parameter(torch.ones(n_neurons)) #multiplicative component / rescale
    self.beta = nn.Parameter(torch.zeros(n_neurons)) #additive component / shift

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    s_input = input.shape # n_batch x n_neurons
    if s_input[1] != self.n_neurons:
        raise Exception(f"Number of neurons does not match! Expected: {self.n_neurons}, Received: {s_input[1]}")

    #compute mean
    mu = input.mean(dim=0)
    s_mu = mu.shape
    #compute variance
    var = input.var(dim=0, unbiased=False)
    s_var = var.shape

    #normalize
    in_norm = (input - mu) / (var + self.epsilon).sqrt()

    #scale and shift
    out = self.gamma * in_norm + self.beta

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization


      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    s_input = input.shape  # n_batch x n_neurons
    #if s_input[1] != self.n_neurons:
    #    raise Exception(f"Number of neurons does not match! Expected: {self.n_neurons}, Received: {s_input[1]}")

    # compute mean
    mu = input.mean(dim=0)
    s_mu = mu.shape
    # compute variance
    var = input.var(dim=0, unbiased=False)
    s_var = var.shape

    # normalize
    denominator = (var + eps).sqrt()
    x_hat = (input - mu) / denominator

    # scale and shift
    out = gamma * x_hat + beta

    #store constants in object
    ctx.epsilon = eps #Store constant non-tensor objects via ctx.constant=myconstant
    #Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
    #ctx.save_for_backward(gamma, beta, x_hat, mu, var, denominator)
    ctx.save_for_backward(gamma, x_hat, denominator)

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass

      grad_output: the grad of the output we receive from the previous layer
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    batch_size, dim = grad_output.shape

    #gamma, beta, x_hat, mu, var, denominator = ctx.saved_tensors
    gamma, x_hat, denominator = ctx.saved_tensors
    epsilon = ctx.epsilon

    # order: input, gamma, beta
    #ctx.needs_input_grad


    #compute gradient for gamma
    if ctx.needs_input_grad[1]:
        grad_gamma = torch.sum(torch.mul(grad_output, x_hat), dim=0)
    else:
        grad_gamma = None

    #compute gradient for beta
    if ctx.needs_input_grad[2]:
        grad_beta = torch.sum(grad_output, dim=0)
    else:
        grad_beta = None

    # compute gradient for input x
    if ctx.needs_input_grad[0]:
        grad_xhat = torch.mul(grad_output, gamma)
        #grad_xhat = grad_output * gamma

        grad_input = 1/batch_size * 1/denominator * \
                     (batch_size*grad_xhat - torch.sum(grad_xhat, dim=0) - x_hat * torch.sum(grad_xhat * x_hat, dim=0))
    else:
        grad_input = None

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    #save variables in object
    self.epsilon = eps
    self.n_neurons = n_neurons

    self.gamma = nn.Parameter(torch.ones(n_neurons)) #multiplicative component / rescale
    self.beta = nn.Parameter(torch.zeros(n_neurons)) #additive component / shift

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    """

    batch_size, n_neurons = input.shape
    #Check for the correctness of the shape of the input tensor.
    if n_neurons != self.n_neurons:
        raise Exception(f"Number of neurons does not match! Expected: {self.n_neurons}, Received: {n_neurons}")

    #Instantiate a CustomBatchNormManualFunction.
    batch_normalization = CustomBatchNormManualFunction() #BN

    #  Call it via its .apply() method
    out = batch_normalization.apply(input, self.gamma, self.beta, self.epsilon)

    return out
