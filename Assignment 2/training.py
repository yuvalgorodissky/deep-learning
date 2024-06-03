import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import SiameseNetwork
from data_preprocessing import get_dataloader
from utility import split_dataloader,plot_losses,calc_accuracy
from tqdm import tqdm


def train_siamese_network(train_dataloader,dev_dataloader, epochs, optimizer, model, device):
    print(f"Training on device: {device}")

    # Create DataLoader
    losses = []
    # Training loop
    model.train()
    for epoch in range(epochs):
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            total_loss = 0
            for images, labels in train_dataloader:
                # Move data to the appropriate device
                img1, img2, labels = images[0].to(device), images[1].to(device), labels.to(device)
                # Zero the gradients before running the backward pass.
                optimizer.zero_grad()
                # Forward pass: Compute predicted outputs by passing images to the model
                output = model(img1, img2)
                # Calculate the loss
                loss = F.binary_cross_entropy(output, labels.unsqueeze(1).float())
                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # Perform a single optimization step (parameter update)
                optimizer.step()
                # Record the loss
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                average_loss = total_loss / len(train_dataloader)
                losses.append(average_loss)
        if (epoch + 1) % 5 == 0:
            all_predictions, all_labels= model.get_pred_labels(dev_dataloader,device)
            accuracy = calc_accuracy(all_predictions ,all_labels)
            print(f'\n dev Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(train_dataloader)}')
    return model, losses


# Example of a call to train_siamese_network

