import datetime
import math


import torch
from torch import tensor
from tqdm import tqdm
from backward_propagation import *
from forward_propagation import *
import matplotlib.pyplot as plt



def L_layer_model(train_dataloader, val_dataloader, layers_dims, learning_rate, num_iterations, batch_norm,
                  use_l2=False, epislon=1e-4):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function,
     and the final layer will apply the softmax activation function. The size of the output layer should be equal to the number of labels in the data.
     Please select a batch size that enables your code to run well (i.e. no memory overflows while still running relatively fast).
    """
    params = initialize_parameters(layers_dims)
    pbar = tqdm(range(num_iterations))
    losses = []
    acc_dev = []
    step_num = 0
    costs = []
    for i in pbar:
        for minibatch_X, minibatch_Y in train_dataloader:
            minibatch_X = minibatch_X.T
            minibatch_Y = torch.nn.functional.one_hot(minibatch_Y, num_classes=layers_dims[-1]).T
            AL, caches = L_model_forward(minibatch_X, params, use_batchnorm=batch_norm)
            cost = compute_cost(AL, minibatch_Y, params, use_l2=use_l2, epsilon=epislon)
            grads = L_model_backward(AL, minibatch_Y, caches, use_l2=use_l2, epsilon=epislon)
            params = update_parameters(params, grads, learning_rate)
            costs.append(cost)
            step_num += 1
            if (step_num + 1) % 100 == 0:
                step_loss = sum(costs) / 100
                losses.append(step_loss)
                costs = []
                acc = Predict(val_dataloader, params, batch_norm)
                acc_dev.append(acc)
                pbar.set_description(f"Loss after step {step_num + 1}: {step_loss:.4f} | Dev accuracy: {acc:.4f}")
                if len(acc_dev) > 1 and abs(acc_dev[-1] - acc_dev[-2]) < 1e-5:
                    print(f"Early stopping at epoch {i + 1}")
                    return params, losses

    return params, losses


def Predict(test_dataloader, parameters, batch_norm):
    """
    This function is used to predict the results of an L-layer neural network.
    """
    total_correct = 0
    total_samples = 0
    for minibatch_X, minibatch_Y in test_dataloader:
        minibatch_X = minibatch_X.T
        minibatch_Y = torch.nn.functional.one_hot(minibatch_Y, num_classes=10)
        AL, caches = L_model_forward(minibatch_X, parameters, use_batchnorm=batch_norm)
        predictions = torch.argmax(AL, dim=0)
        correct_predictions = predictions == torch.argmax(minibatch_Y, dim=1)  # Compare max indices
        total_correct += correct_predictions.sum().item()
        total_samples += minibatch_Y.shape[0]  # Assuming minibatch_Y is [num_classes, batch_size]

    accuracy = total_correct / total_samples

    return accuracy





def save_cost_graph(costs, filename, learning_rate, batch_norm, use_l2, epsilon, num_iterations, train_batch_size, test_batch_size, train_acc, dev_acc, test_acc):
    """
    Saves a graph of the costs and accuracies over time to a file.
    """

    fig, ax1 = plt.subplots()

    # Plotting the cost on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(costs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis to plot the accuracies
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(train_acc, label='Train Accuracy', color='blue', linestyle='--')
    ax2.plot(dev_acc, label='Dev Accuracy', color='green', linestyle='-.')
    ax2.plot(test_acc, label='Test Accuracy', color='purple', linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15))

    # Adding a title and adjusting the layout
    plt.title(f"Training Overview with learning rate {learning_rate}, batch norm {batch_norm}, use L2 {use_l2},\n epsilon {epsilon}, iterations {num_iterations}, train batch size {train_batch_size}, test batch size {test_batch_size}")
    fig.tight_layout()

    # Saving the graph to a file with a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{filename}_{current_time}.png")

    # Display the graph
    plt.show()

