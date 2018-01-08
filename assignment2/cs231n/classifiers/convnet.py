from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    (conv_relu - 2x2 max pool)X2 - affine_bn_relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=(64,128), filter_size=(5,3),
                 hidden_dim=200, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
       
        param = C
        # convolutional layer
        for i in range(2):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(num_filters[i],param,filter_size[i],filter_size[i])
            param = num_filters[i]
            
            self.params['b'+str(i+1)] = np.zeros(num_filters[i])
            
        # FC layer
        self.params['W3'] = weight_scale * np.random.randn(num_filters[1]*(H//4)*(W//4),hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['gamma'] = np.ones(hidden_dim)
        self.params['beta'] = np.zeros(hidden_dim)
        
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b4'] = np.zeros(num_classes)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        self.bn_param = {'mode': 'train'}
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        self.bn_param['mode'] = mode
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma, beta = self.params['gamma'], self.params['beta']
        
        # pass conv_param to the forward pass for the convolutional layer
        conv_params = [{'stride': 1, 'pad': 2},{'stride': 1, 'pad': 1}]

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride':2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        c_r_p_out1, c_r_p_cache1 = conv_relu_pool_forward(X, W1, b1, conv_params[0], pool_param)
        c_r_p_out2, c_r_p_cache2 = conv_relu_pool_forward(c_r_p_out1, W2, b2, conv_params[1], pool_param)
        
        a_b_r_out, a_b_r_cache = affine_bn_relu_forward(c_r_p_out2, W3, b3, gamma, beta, self.bn_param)
        affine_out, affine_cache = affine_forward(a_b_r_out, W4, b4)
        
        scores = affine_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, daffine_out = softmax_loss(affine_out,y)
        loss += 0.5 * self.reg * (np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4))

        da_b_r_out, dW4, db4 = affine_backward(daffine_out, affine_cache)   
        dc_r_p_out2, dW3, db3, dgamma, dbeta = affine_bn_relu_backward(da_b_r_out, a_b_r_cache)
        dc_r_p_out1, dW2, db2 = conv_relu_pool_backward(dc_r_p_out2, c_r_p_cache2)
        dx, dW1, db1 = conv_relu_pool_backward(dc_r_p_out1, c_r_p_cache1)
        
        grads['W1'] = dW1 + self.reg * W1
        grads['W2'] = dW2 + self.reg * W2
        grads['W3'] = dW3 + self.reg * W3
        grads['W4'] = dW4 + self.reg * W4
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        grads['b4'] = db4
        grads['gamma'] = dgamma
        grads['beta'] = dbeta

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
