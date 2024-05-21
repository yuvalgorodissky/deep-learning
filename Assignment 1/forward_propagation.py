import torch
from torch import tensor


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)  # Number of layers in the network
    for l in range(1, L):
        # He initialization for weights: sqrt(2/prev_layer_size)
        parameters['W' + str(l)] = torch.randn(layer_dims[l], layer_dims[l - 1]) * torch.sqrt(
            torch.tensor(2.0 / layer_dims[l - 1]))
        # Initialize biases to zero
        parameters['b' + str(l)] = torch.zeros(layer_dims[l], 1)

    return parameters


def linear_forward(A, W, b):
    Z = torch.mm(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache


def softmax(Z):
    expZ = torch.exp(Z)
    A = expZ / torch.sum(expZ, dim=0)
    activation_cache = Z
    return A, activation_cache


def relu(Z):
    A = torch.max(tensor(0), Z)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = {"linear_cache": linear_cache, "activation_cache": activation_cache}
    return A, cache


def apply_batchnorm(A):
    m = A.size()[1]
    mu = torch.sum(A, dim=1, keepdim=True) / m  # Keep dimension to ensure mu is [features, 1]
    sigma_square = torch.sum((A - mu) ** 2, dim=1, keepdim=True) / m  # Also keep dimension here
    sigma = torch.sqrt(sigma_square)
    epsilon = 1e-8
    NA = (A - mu) / torch.sqrt(sigma_square + epsilon)
    return NA


def L_model_forward(X, parameters, use_batchnorm):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y, parameters=None, epsilon=1e-4, use_l2=False):
    m = Y.size()[0]
    cost = -torch.sum(Y * torch.log(AL+1e-8)) / m
    if use_l2:
        L2_cost = 0
        for key in parameters:
            if key[0] == 'W':
                L2_cost += torch.sum(parameters[key] ** 2)
        L2_cost = (epsilon / 2) * L2_cost
        cost += L2_cost

    return cost
