import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]

  for i in xrange(N):
      #We consider samples column-wise, sample is 1xD
      sample = X[i,:]

      scores = sample.dot(W)
      max_score = np.argmax(scores)

      #To avoid numerical instability, now our score values will lay
      #between [-max_score, 0]

      scores -= max_score
      exp_sum = np.sum(np.exp(scores))

      #Loss is updated for each sample and normalized
      #see course notes for the numerical stability of this formula

      loss += -scores[y[i]] + np.log(exp_sum)

      #Here comes the harsh stuff.
      #We have to update dW for each class
      #First we consider if we are grading the correct class, in that
      #case we add minus one, then we add our score value.
      #The score value is raised to e then normalized over
      #the sum of exponentials

      #Anyway the result of the expression between parentheses is just
      # a scalar so, sample has size 1xD and dW[:,j] Dx1

      exp_sum = np.sum(np.exp(scores))

      for j in xrange(C):
          dW[:,j] += sample * (-1 * (j==y[i]) + np.exp(scores[j])/exp_sum )

  #Loss and dW are normalized over the number of training examples

  loss /= N
  dW /= N

  #regularization, for loss is l2

  loss += reg*np.sum(W**2) / 2
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
