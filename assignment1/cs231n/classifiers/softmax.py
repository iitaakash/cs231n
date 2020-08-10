from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dim = X.shape[1]
    loss = 0.0
    
    grad_W_reg = 2 * reg * W
    
    grad_N = -1/num_train 
    
    for i in range(num_train):
        scores = np.exp(X[i].dot(W))
        correct_class_score = scores[y[i]]
        sum_val = np.sum(scores)
        likelihood = correct_class_score / sum_val
        cur_loss = -1 * np.log(likelihood)
        loss += cur_loss
        
        grad_loss = (-1/num_train) * (1/likelihood)
        
        update_denom = -1 * ((grad_loss * correct_class_score) / np.square(sum_val))
        
        update_numer = (grad_loss / sum_val) 
        
        for j in range(num_classes):
            dW[:,j] += update_denom * scores[j] * X[i,:].reshape(num_dim,)
        
        dW[:,y[i]] +=  update_numer * scores[y[i]] * X[i,:].reshape(num_dim,)
            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    dW += grad_W_reg
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    loss = 0.0;
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dim = X.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.exp(X.dot(W))
    likelihood_all = scores / np.sum(scores, axis = 1).reshape(num_train,1)
    loss += -1 * np.sum(np.log(likelihood_all[np.arange(num_train), y]))
    
    condition = np.zeros_like(scores)
    likelihood_all[np.arange(num_train), y] += -1;
    
    dW += (X.T).dot(likelihood_all) 
    
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
