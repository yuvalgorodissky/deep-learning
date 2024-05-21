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
    train_dataloader, val_dataloader, test_dataloader = load_dataset(train_batch_size, test_batch_size)
    # Train the model
    params, losses = L_layer_model(train_dataloader, val_dataloader, layers, learning_rate, num_iterations,
                                   batch_norm=batch_norm, use_l2=use_l2, epislon=epislon)
    save_cost_graph(losses, "losses", learning_rate, batch_norm, use_l2, epislon, num_iterations, train_batch_size,
                    test_batch_size)
    acc = Predict(test_dataloader, params, batch_norm)
    print(f"Test accuracy: {acc:.4f}")


def main():
    layers = [28 * 28, 20, 7, 5, 10]
    batch_norms = [False, True]
    learning_rate = 0.009
    uses_l2 =  [False, True]
    epislon = 1e-4
    nums_iterations = [50, 100, 300]
    train_batch_size = 64
    test_batch_size = 16
    for batch_norm in batch_norms:
        for use_l2 in uses_l2:
            for num_iterations in nums_iterations:
                train(layers, batch_norm, learning_rate, train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                  num_iterations=num_iterations, use_l2=use_l2, epislon=epislon)


if __name__ == "__main__":
    main()
