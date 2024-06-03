import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime


def split_dataloader(dataloader, splits=0.8):
    train_size = int(splits * len(dataloader.dataset))
    test_size = len(dataloader.dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=dataloader.batch_size)
    return train_loader, test_loader

def plot_losses(losses,path):
    ##get the current time
    now = datetime.datetime.now()
    ##plot the losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.savefig(path + 'losses_' + str(now) + '.png')
    plt.show()


def calc_accuracy(labels, preds):
    return 100*(labels == preds.unsqueeze(1)).sum().item() / len(labels)
