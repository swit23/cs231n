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
  num_train, num_classes = X.shape[0], W.shape[1]
  
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
        scores = X[i].dot(W)

        #subtract the largest score for regularization purposes
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        
        
        correct_class_score = exp_scores[y[i]]
        loss -= np.log(correct_class_score / np.sum(exp_scores))
        
        #update gradient
        dW[:,y[i]] += X[i] * ((correct_class_score / np.sum(exp_scores)) - 1) 
        for j in range(num_classes):
            if j == y[i]:
                continue
            else:
                dW[:,j] += X[i] * exp_scores[j] / np.sum(exp_scores)
   
   
            
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss / num_train + reg * np.sum(W * W), dW / num_train + 2 * reg * W
 

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train, num_classes = X.shape[0], W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  
  #subtract regularization factor
  scores -= np.max(scores, axis=1).reshape(num_train, -1)
   
  exp_scores = np.exp(scores)
  
  #break into steps for clarity
  class_sums = np.sum(exp_scores, axis=1)
  correct_class_scores = exp_scores[np.arange(num_train), y] 

  L = -np.log(correct_class_scores / class_sums)
  L /= num_train

  loss += np.sum(L) + reg * np.sum(W * W)

  #loss = np.sum( -np.log(exp_scores[np.arange(num_train), y] * 1 / np.sum(exp_scores, axis=1))) / num_train

  exp_scores /= class_sums.reshape(num_train, -1)
  
  m = exp_scores.T.dot(X).T
  exp_scores.fill(0)

  #select X[i] corresponding to y[i]
  exp_scores[np.arange(num_train), y] = 1
  n = exp_scores.T.dot(X).T

  dW = m - n
  dW /= num_train
  
    
    
 
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

