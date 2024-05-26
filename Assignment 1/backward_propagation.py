import torch
from forward_propagation import *


def linear_backward(dZ, cache, epsilon=1e-4, use_l2=False):
    A_prev, W, b = cache
    m = A_prev.size()[1]  # Number of examples

    # Compute gradients of the cost with respect to W and b
    dW = torch.mm(dZ, A_prev.T) / m
    db = torch.sum(dZ, dim=1, keepdim=True) / m
    dA_prev = torch.mm(W.T, dZ)
    # Add the gradient of the regularization term if L2 regularization is used
    if use_l2:
        dW += epsilon * W / m

    return dA_prev, dW, db

def softmax_backward(dA, cache):
    Z = cache
    S = torch.exp(Z) / torch.sum(torch.exp(Z), dim=0)
    dZ = dA * S * (1 - S)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = dA.clone()
    dZ[Z <= 0] = 0

    return dZ


def linear_activation_backward(dA, cache, activation, use_l2=False, epsilon=1e-4):
    linear_cache, activation_cache = cache["linear_cache"], cache["activation_cache"]
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache, epsilon=epsilon, use_l2=use_l2)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, use_l2=False, epsilon=1e-4):
    """
    Implement the backward propagation process for the entire network.
    """
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    # For softmax, we directly compute dAL from AL and Y (output of the last layer)
    dAL = - (Y / (AL + 1e-8)) + (1 - Y) / (1 - AL + 1e-8)
    grads["dA" + str(L)] = dAL

    # Lth layer (softmax -> linear) gradients. Inputs: "AL, Y, caches".
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
        grads["dA" + str(L)], current_cache, "softmax", use_l2=use_l2, epsilon=epsilon)

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        # lth layer: (Relu -> Linear) gradients.
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu", use_l2=use_l2, epsilon=epsilon)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters
