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


def plot_losses(losses, path):
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
    return 100 * (labels == preds.unsqueeze(1)).sum().item() / len(labels)


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
        headers = ['Optimizer', 'Dropout', 'BatchNorm', 'BatchSize', 'Test Accuracy', 'Train Accuracy', 'Val Accuracy',
                   'Time']
        writer.writerow(headers)

        # Write data rows
        for key, value in result_dict.items():
            optimizer_name, dropout, batch_norm, batch_size = key
            test_accuracy = round(value['test_accuracy'], 3)
            train_accuracy = round(value['train_accuracy'], 3)
            val_accuracy = round(value['val_accuracy'], 3)
            time = value['time']
            row = [optimizer_name, dropout, batch_norm, batch_size, test_accuracy, train_accuracy, val_accuracy, time]
            writer.writerow(row)


def collect_samples_indices(dataloader, preds):
    # Initialize the dictionary with categories of interest
    samples_dict = {
        'true_positive': [],  # Predicted 1, actual 1
        'false_positive': [],  # Predicted 1, actual 0
        'true_negative': [],  # Predicted 0, actual 0
        'false_negative': []  # Predicted 0, actual 1
    }

    for i, pred in enumerate(preds):
        label = dataloader.dataset[i][1]
        image1 = dataloader.dataset[i][0][0][0]
        image2 = dataloader.dataset[i][0][1][0]
        if pred == 1 and label == 1:
            option = 'true_positive'
        elif pred == 1 and label == 0:
            option = 'false_positive'
        elif pred == 0 and label == 0:
            option = 'true_negative'
        else:
            option = 'false_negative'
        samples_dict[option].append((image1, image2))

    return samples_dict


def save_samples_to_file(samples_dict):
    # We will display only 2 pairs per key, with each pair having 2 images side by side
    nrows_per_key = 2  # 2 pairs
    ncols_per_pair = 2  # 2 images per pair

    # Create a large single figure for all samples
    total_keys = len(samples_dict)
    # Adjusting the size to make each image larger
    fig, axs = plt.subplots(nrows_per_key * total_keys, ncols_per_pair, figsize=(12, 4 * total_keys), dpi=100)

    current_row = 0
    for key, value in samples_dict.items():
        if len(value) >= 2:
            # Place a title over the first pair of each key's section
            axs[current_row, 0].set_title(key)

            # Limit the number of pairs displayed per key to 2 pairs
            displayed_pairs = value[:2]
            for pair in displayed_pairs:
                # Display the first image in the pair
                axs[current_row, 0].imshow(pair[0], cmap='gray')
                axs[current_row, 0].axis('off')

                # Display the second image in the pair
                axs[current_row, 1].imshow(pair[1], cmap='gray')
                axs[current_row, 1].axis('off')

                current_row += 1

    # Adjust the layout to make room for the main title and ensure tight packing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Add a main title to the figure
    plt.suptitle('All Samples')
    # Save the entire figure as a single PNG file
    plt.savefig('all_samples.png')
    plt.close(fig)

