import torch
from model import SiameseNetwork
from training import train_siamese_network
from utility import plot_losses, split_dataloader, calc_accuracy, export_result_dict, collect_samples_indices, \
    save_samples_to_file
from data_preprocessing import get_dataloader
from torch.optim.lr_scheduler import StepLR


def main():
    # Path setup
    images_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/lfw2"
    text_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/pairsDevTrain.txt"
    test_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/pairsDevTest.txt"
    base_writer_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/runs"  # TensorBoard logs directory
    optimizers = ["Adam", "SGD", "RMSprop"]
    dropouts = [0.2, 0.5, 0]
    batch_norms = [True, False]
    batch_sizes = [8, 32, 128]

    optimizers = ["Adam"]
    dropouts = [0.2]
    batch_norms = [True]
    batch_sizes = [32]

    results_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for optimizer_name in optimizers:
        for dropout in dropouts:
            for batch_norm in batch_norms:
                for batch_size in batch_sizes:
                    writer_path = f"{base_writer_path}/{optimizer_name}_{dropout}_{batch_norm}_{batch_size}"
                    print(
                        f"Running with optimizer: {optimizer_name}, dropout: {dropout}, batch_norm: {batch_norm}, batch_size: {batch_size}")
                    train_dataloader, dev_dataloader = split_dataloader(
                        get_dataloader(text_path, images_path, batch_size=batch_size))
                    test_dataloader = get_dataloader(test_path, images_path, batch_size=batch_size, is_train=False)

                    model = SiameseNetwork(dropout=dropout, batch_norm=batch_norm).to(device)

                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                    elif optimizer_name == "SGD":
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)
                    else:
                        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)
                    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                    model, time = train_siamese_network(model, train_dataloader, dev_dataloader, epochs=2,
                                                        optimizer=optimizer,
                                                        scheduler=scheduler, device=device, writer_path=writer_path)

                    test_labels, test_predictions = model.get_pred_labels(test_dataloader, device)
                    test_accuracy = calc_accuracy(test_predictions, test_labels)

                    train_labels, train_predictions = model.get_pred_labels(train_dataloader, device)
                    train_accuracy = calc_accuracy(train_predictions, train_labels)

                    val_labels, val_predictions = model.get_pred_labels(dev_dataloader, device)
                    val_accuracy = calc_accuracy(val_predictions, val_labels)

                    print(f"Test accuracy: {test_accuracy:.2f}% time - {time}")
                    results_dict[(optimizer_name, dropout, batch_norm, batch_size)] = {"test_accuracy": test_accuracy,
                                                                                       "train_accuracy": train_accuracy,
                                                                                       "val_accuracy": val_accuracy,
                                                                                       "time": time}
                    sample_dict = collect_samples_indices(test_dataloader, test_predictions)
                    save_samples_to_file(sample_dict)

    export_result_dict(results_dict)


if __name__ == "__main__":
    main()
