import torch
from torch import nn

from model import SiameseNetwork, SiameseBaseNetwork
from training import train_siamese_network
from utility import plot_losses, calc_accuracy, export_result_dict, save_model, load_model

from data_preprocessing import get_dataloader
from torch.optim.lr_scheduler import StepLR
from evaluation import collect_samples_indices, save_samples_to_file, confusion_matrix
import torch.nn.functional as F


def main():
    # Path setup
    images_path = "data/lfw2"
    train_path = "data/pairsDevTrain.txt"
    test_path = "data/pairsDevTest.txt"
    exp_number = 11


    base_writer_path = f"runs/exp{exp_number}"
    optimizers = ["Adam", "RMSprop"]
    dropouts = [0.2, 0]
    batch_norms = [True, False]
    batch_sizes = [8, 32, 128]
    loss_fns_names = ["binary_cross_entropy", "mse_loss"]
    loss_fns = [F.binary_cross_entropy, F.mse_loss]
    use_augmentations = [1, 10]
    number_of_models = len(optimizers) * len(dropouts) * len(batch_norms) * len(batch_sizes) * len(loss_fns) * len(
        use_augmentations)
    print(f"Number of models: {number_of_models}")

    epochs = 50
    each_checkpoints = 5
    num_of_checkpoints = epochs // each_checkpoints

    results_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_number = 0
    for optimizer_name in optimizers:
        for dropout in dropouts:
            for batch_norm in batch_norms:
                for batch_size in batch_sizes:
                    for loss_fn_name, loss_fn in zip(loss_fns_names, loss_fns):
                        for use_augmentation in use_augmentations:
                            model_number += 1
                            train_dataloader, dev_dataloader = get_dataloader(train_path, images_path,
                                                                              batch_size=batch_size,
                                                                              use_augmentation=use_augmentation,
                                                                              splits=1)
                            test_dataloader = get_dataloader(test_path, images_path, batch_size=batch_size,
                                                             use_augmentation=0)

                            model = SiameseNetwork(dropout=dropout, batch_norm=batch_norm).to(device)

                            if optimizer_name == "Adam":
                                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                            elif optimizer_name == "SGD":
                                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)
                            else:
                                optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)
                            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                            for epoch in range(1, num_of_checkpoints):
                                model_number += 1
                                writer_path = f"{base_writer_path}/{optimizer_name}_{dropout}_{batch_norm}_{batch_size}_{loss_fn_name}_{use_augmentation}_epoch_{epoch * each_checkpoints}"
                                print(
                                    f"|| Running model number {model_number}/{number_of_models}  || \noptimizer: {optimizer_name}, dropout: {dropout}, batch_norm: {batch_norm}, batch_size: {batch_size}, loss_fn: {loss_fn_name}, use_augmentation: {use_augmentation}_epoch_{epoch * each_checkpoints}")

                                model, time = train_siamese_network(model, train_dataloader, dev_dataloader,
                                                                    epochs=each_checkpoints, loss_fn=loss_fn,
                                                                    optimizer=optimizer,
                                                                    scheduler=scheduler, device=device,
                                                                    writer_path=writer_path)

                                test_labels, test_predictions = model.get_pred_labels(test_dataloader, device)
                                test_accuracy = calc_accuracy(test_predictions, test_labels)

                                train_labels, train_predictions = model.get_pred_labels(train_dataloader, device)
                                train_accuracy = calc_accuracy(train_predictions, train_labels)

                                val_labels, val_predictions = model.get_pred_labels(dev_dataloader, device)
                                val_accuracy = calc_accuracy(val_predictions, val_labels)
                                print(f"Test accuracy: {test_accuracy:.2f}% time - {time}")
                                results_dict[(
                                optimizer_name, dropout, batch_norm, batch_size, loss_fn_name, use_augmentation,
                                epoch * each_checkpoints)] = {"test_accuracy": test_accuracy,
                                                              "train_accuracy": train_accuracy,
                                                              "val_accuracy": val_accuracy,
                                                              "time": time}

                                save_model(model, f"models/model_{model_number}_cp{each_checkpoints * epoch}.pt")

    export_result_dict(results_dict)


if __name__ == "__main__":
    main()
