import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from trainer import *


def load_dataset(train_batch_size=64, test_batch_size=16):
    # Define the transformation to convert images to tensors and flatten them
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to [C, H, W] format where H=W=28 and C=1
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])

    # Load the training and testing datasets
    full_trainset = datasets.MNIST(root='./train', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    trainloader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size)
    valloader = DataLoader(valset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.MNIST(root='./test', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, shuffle=True, batch_size=test_batch_size)

    return trainloader, valloader, testloader


def train(layers, batch_norm, learning_rate, num_iterations=80, train_batch_size=64, test_batch_size=16, use_l2=False,
          epislon=1e-4):
    start_dt = datetime.datetime.now()
    train_dataloader, val_dataloader, test_dataloader = load_dataset(train_batch_size, test_batch_size)
    # Train the model
    params, losses, num_iterations = L_layer_model(train_dataloader, val_dataloader, layers, learning_rate,
                                                   num_iterations,
                                                   batch_norm=batch_norm, use_l2=use_l2, epislon=epislon)
    train_acc = Predict(train_dataloader, params, batch_norm)
    print(f"Train accuracy: {train_acc:.4f}")
    dev_acc = Predict(val_dataloader, params, batch_norm)
    print(f"Dev accuracy: {dev_acc:.4f}")
    # Predict the test set
    acc_test = Predict(test_dataloader, params, batch_norm)
    print(f"Test accuracy: {acc_test:.4f}")
    # Calculate the total time taken in mm:ss
    end_dt = datetime.datetime.now()
    time_diff = end_dt - start_dt
    minutes = time_diff.seconds // 60
    seconds = time_diff.seconds % 60
    total_time = f"{minutes:02d}:{seconds:02d}"
    save_cost_graph(losses, "losses", learning_rate, batch_norm, use_l2, epislon, num_iterations, train_batch_size,
                    test_batch_size, train_acc, dev_acc, acc_test, total_time)
    return params


def compare_weights(params_without_l2, params_with_l2):
    """
    Compares the weights of two models, one trained with L2 regularization and one without.
    It plots the distribution of weights for each model side-by-side for each layer and displays the average weight value.

    :param params_without_l2: Parameters from the model trained without L2 regularization.
    :param params_with_l2: Parameters from the model trained with L2 regularization.
    """
    num_layers = sum(1 for key in params_without_l2.keys() if 'W' in key)
    fig, axes = plt.subplots(nrows=num_layers, ncols=2, figsize=(12, num_layers * 3))

    i = 0
    for key in params_without_l2:
        if 'W' in key:  # Assuming the weights are denoted by 'W'
            W_without_l2 = params_without_l2[key]
            W_with_l2 = params_with_l2[key]
            mean_without_l2 = torch.mean(W_without_l2).item()
            mean_with_l2 = torch.mean(W_with_l2).item()

            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
            ax1.hist(W_without_l2.detach().numpy().flatten(), bins=50, alpha=0.5, color='blue')
            ax2.hist(W_with_l2.detach().numpy().flatten(), bins=50, alpha=0.5, color='green')
            ax1.set_title(f'Without L2: {key} (Avg: {mean_without_l2:.2e})')
            ax2.set_title(f'With L2: {key} (Avg: {mean_with_l2:.2e})')
            ax1.set_xlabel('Weight Value')
            ax1.set_ylabel('Frequency')
            ax2.set_xlabel('Weight Value')
            ax2.set_ylabel('Frequency')

            # Add text inside the plot with average values
            ax1.text(0.05, 0.95, f'Avg: {mean_without_l2:.2e}', transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax2.text(0.05, 0.95, f'Avg: {mean_with_l2:.2e}', transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            i += 1

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"weights_comparison-{current_time}.png")
    plt.show()


def main():
    global params_without_l2, params_with_l2
    layers = [28 * 28, 20, 7, 5, 10]
    batch_norms = [False, True]
    learning_rate = 0.009
    uses_l2 = [False, True]
    epislon = 1e-4
    nums_iterations = [200]
    train_batch_size = 64
    test_batch_size = 16

    for use_l2 in uses_l2:
        for batch_norm in batch_norms:
            for num_iterations in nums_iterations:
                if batch_norm and use_l2:
                    continue
                if not batch_norm and not use_l2:
                    params_without_l2 = train(layers, batch_norm, learning_rate, num_iterations, train_batch_size,
                                              test_batch_size, use_l2, epislon)
                elif not batch_norm and use_l2:
                    params_with_l2 = train(layers, batch_norm, learning_rate, num_iterations, train_batch_size,
                                           test_batch_size, use_l2, epislon)
                else:
                    _ = train(layers, batch_norm, learning_rate, num_iterations, train_batch_size, test_batch_size,
                              use_l2, epislon)

    compare_weights(params_without_l2, params_with_l2)


if __name__ == "__main__":
    main()
