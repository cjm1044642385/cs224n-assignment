# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    # a1 = z1 = X
    z2 = np.dot(X, W1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W2) + b2
    a3 = softmax(z3)
    cost = -np.sum(np.log(np.sum(a3 * labels, 1))) # a3 * labels是矩阵点乘
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # 作业题2（b），其实就是给出了delta3的求法。
    delta3 = a3 - labels
    # 按照 https://www.cnblogs.com/royhoo/p/9149172.html 给出的公式计算。不过在代码里面，参数矩阵的行列，与文章里面刚好相反。
    delta2 = sigmoid_grad(a2) * np.dot(delta3, W2.T)
    gradb2 = np.sum(delta3, 0)
    gradb1 = np.sum(delta2, 0)
    gradW2 = np.zeros([H, Dy])
    gradW1 = np.zeros([Dx, H])
    for i in range(0, X.shape[0]):
        gradW2 = gradW2 + np.dot(a2[i].reshape(-1, 1), delta3[i].reshape(-1, 1).T)
        gradW1 = gradW1 + np.dot(X[i].reshape(-1, 1), delta2[i].reshape(-1, 1).T)

    ### END YOUR CODE
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    cost = forward_backward_prop(data, labels, params, dimensions)
    print cost
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
