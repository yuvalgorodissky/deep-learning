import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
import csv

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




def export_result_dict(result_dict):
    # Format the current date and time as a string that is safe for filenames
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the path with the formatted datetime
    filename = 'results_' + now + '.csv'

    # Open the file for writing
    with open(filename, 'w', newline='') as f:
        # Create a CSV writer object
        writer = csv.writer(f)

        # Write the header row
        headers = ['Optimizer', 'Dropout', 'BatchNorm', 'BatchSize', 'Test Accuracy', 'Train Accuracy', 'Val Accuracy', 'Time']
        writer.writerow(headers)

        # Write data rows
        for key, value in result_dict.items():
            optimizer_name, dropout, batch_norm, batch_size = key
            test_accuracy = round(value['test_accuracy'], 3)
            train_accuracy = round(value[train_accuracy], 3)
            val_accuracy = round(value['val_accuracy'], 3)
            time= value['time']
            row = [optimizer_name, dropout, batch_norm, batch_size, test_accuracy, train_accuracy, val_accuracy, time]
            writer.writerow(row)


def collect_samples_indices(dataloader, preds, labels):
    # Initialize the dictionary with categories of interest
    samples_dict = {
        'true_positive': [],  # Predicted 1, actual 1
        'false_positive': [],  # Predicted 1, actual 0
        'true_negative': [],  # Predicted 0, actual 0
        'false_negative': []  # Predicted 0, actual 1
    }

    # Keep a global index across all batches
    global_index = 0

    # Loop through batches
    for batch_images, batch_preds, batch_labels in zip(dataloader, preds, labels):
        for pred, label in zip(batch_preds, batch_labels):
            if pred == 1 and label == 1:
                samples_dict['true_positive'].append(global_index)
            elif pred == 1 and label == 0:
                samples_dict['false_positive'].append(global_index)
            elif pred == 0 and label == 0:
                samples_dict['true_negative'].append(global_index)
            elif pred == 0 and label == 1:
                samples_dict['false_negative'].append(global_index)

            global_index += 1

    return samples_dict
