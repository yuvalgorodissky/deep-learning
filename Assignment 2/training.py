import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import SiameseNetwork
from data_preprocessing import get_dataloader
from utility import plot_losses, calc_accuracy
from tqdm import tqdm

import datetime
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard


def train_siamese_network(model, train_dataloader, dev_dataloader, epochs, loss_fn, optimizer, scheduler, device,
                          writer_path):
    start_dt = datetime.datetime.now()
    writer = SummaryWriter(writer_path)  # Initialize TensorBoard writer
    dev_losses = []
    print(f"Training on device: {device}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            img1, img2, labels = images[0].to(device), images[1].to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = loss_fn(output, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar('Training loss', loss.item(), epoch * len(train_dataloader) + i)  # Log loss

        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}')
        writer.add_scalar('Average Training loss', avg_loss, epoch)  # Log average training loss
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, (images, labels) in enumerate(dev_dataloader):
                    img1, img2, labels = images[0].to(device), images[1].to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(img1, img2)
                    loss = loss_fn(output, labels.unsqueeze(1).float())
                    total_loss += loss.item()
                total_loss = total_loss / len(dev_dataloader)
                dev_losses.append(total_loss)
                writer.add_scalar('Dev loss', total_loss, epoch)  # Log loss
                if len(dev_losses) > 1 and abs(dev_losses[-1] - dev_losses[-2]) < 0.0001:
                    end_dt = datetime.datetime.now()
                    time_diff = end_dt - start_dt
                    total_time = f"{time_diff.seconds // 60:02d}:{time_diff.seconds % 60:02d}"
                    writer.close()  # Close the writer
                    print(f'Early stopping after epoch {epoch + 1}')
                    return model, total_time
            model.train()

    end_dt = datetime.datetime.now()
    time_diff = end_dt - start_dt
    total_time = f"{time_diff.seconds // 60:02d}:{time_diff.seconds % 60:02d}"
    writer.close()  # Close the writer
    return model, total_time
