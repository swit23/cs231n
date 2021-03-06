import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count += 1
        loss += margin
        dW[:,j] += X[i]
    dW[:,y[i]] += - count * X[i]
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  #Correct gradient by diving by num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization term to gradient
  dW += reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train, num_classes = X.shape[0], dW.shape[1]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                        a   #
  #############################################################################
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(X.shape[0], dtype=int), y].reshape(X.shape[0],-1)  
  margins = scores - correct_class_scores + 1

  #for the data point x_i, zero out the score for the correct class y[i]
  margins = np.maximum(margins, 0)
  margins[np.arange(X.shape[0], dtype=int), y] = 0

  loss += np.sum(margins) / X.shape[0]
 
  #add regularization term
  loss += reg * np.sum(W * W)
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  args = margins > 0
  margins[args] = 1
 
  class_weights = -np.sum(margins, axis=1).reshape(X.shape[0], -1)

  #to the jth column, add each point X[i] with nonzero margin on class j
  #m = np.einsum("ij,ik -> kj", margins, X)
  m = margins.T.dot(X).T
    
  margins[args] = 0
  margins[np.arange(num_train, dtype=int), y] = 1
  #for each y in range(num_train), add X[i] weighted by the number of nonzero margins in class y[i] to the column y[i]
  #n = np.einsum("ij,ik -> kj",  margins, class_weights * X)
  n = margins.T.dot(class_weights * X).T
    
  dW += m + n

  dW /= num_train
    
  #add regularization gradient

  dW += 2 * reg * W
 

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
