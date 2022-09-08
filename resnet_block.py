#import tensorflow as tf
import numpy as np
import torch


def LeakyRelu(x):
    return torch.nn.LeakyReLU()(x)
    
def OurRelu(x):

    leak = 0.1
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * abs(x) - f2 * x

def Friend_relu(x):
    x = torch.nn.relu(x)
    b = 255.0 * torch.ones_like(x)
    return min(x, b)

def GroupNorm(x,G=32):
    N, C,H, W = x.shape
    groupNorm = torch.nn.GroupNorm(G, C).cuda()
    result = groupNorm(x)

    return result

def identity_block(X_input, kernel_size, in_filter, out_filters):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        f1, f2, f3 = out_filters
        X_shortcut = X_input
        #first
        conv1 = torch.nn.Conv2d( in_filter, f1,1,1).cuda()
        X = conv1(X_input)
        X = GroupNorm(X)
        X = OurRelu(X)#1X69x69x256

        #second
        conv2 = torch.nn.Conv2d( f1, f2, kernel_size,1,1).cuda()
        X = conv2(X)
        X = GroupNorm(X)
        X = OurRelu(X)

        #third
        conv3 = torch.nn.Conv2d(f2, f3, 1, 1).cuda()

        X = conv3(X)
        X = GroupNorm(X)

        #final step
        add = torch.add(X, X_shortcut)
        add_result = OurRelu(add)
        return add_result

def convolutional_block(X_input, kernel_size, in_filter,
                            out_filters, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis

        f1, f2, f3 = out_filters

        x_shortcut = X_input
        #first
        conv1 = torch.nn.Conv2d(in_filter,f1,1,stride,)
        X = conv1(X_input)
        X = GroupNorm(X)
        X = LeakyRelu(X)

        #second
        conv2 = torch.nn.Conv2d(f1, f2,kernel_size, 1,1 )
        X = conv2(X)
        X = GroupNorm(X)
        X = LeakyRelu(X)

        #third
        conv3 = torch.nn.Conv2d(f2, f3,1, 1)
        X = conv3(X)
        X = GroupNorm(X)

        conv4 = torch.nn.Conv2d(in_filter, f3, 1, stride)
        x_shortcut = conv4(x_shortcut)

        #final
        add = torch.add(x_shortcut, X)
        add_result = LeakyRelu(add)
        return add_result
