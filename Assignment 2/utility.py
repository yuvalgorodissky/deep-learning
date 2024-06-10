import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
import csv
from data_preprocessing import get_transforms

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
        headers = ['Optimizer','Epochs', 'Dropout', 'BatchNorm', 'BatchSize', 'loss_fn_name', 'use_augmentation', 'TestAccuracy',
                   'TrainAccuracy', 'ValAccuracy', 'Time']
        writer.writerow(headers)

        # Write data rows
        for key, value in result_dict.items():
            optimizer_name, dropout, batch_norm, batch_size, loss_fn_name, use_augmentation ,epochs= key
            test_accuracy = round(value["test_accuracy"],3)
            train_accuracy = round(value["train_accuracy"],3)
            val_accuracy =round(value["val_accuracy"],3)
            time = value["time"]
            row = [optimizer_name,epochs, dropout, batch_norm, batch_size, loss_fn_name, use_augmentation, test_accuracy,
                   train_accuracy, val_accuracy, time]
            writer.writerow(row)



from PIL import Image
import numpy as np

def image_transform(image, path):
    transform = get_transforms(True)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.array(image), cmap='gray' if image.mode == 'L' else None)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Apply transformation
    transformed_image = transform(image)

    # Convert tensor back to numpy array for plotting
    transformed_image = transformed_image.permute(1, 2, 0).numpy()
    ax[1].imshow(transformed_image, cmap='gray' if image.mode == 'L' else None)
    ax[1].set_title('Transformed Image')
    ax[1].axis('off')
    plt.savefig(path)

def save_image(path_to_image, path_to_save):
    # Open image in grayscale
    image = Image.open(path_to_image).convert('L')
    image_transform(image, path_to_save)


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    ##load the model from the path saved in save_model function
    model = torch.load(path)
    return model
