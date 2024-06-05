import torch
from torch import nn

from model import SiameseNetwork
from training import train_siamese_network
from utility import plot_losses, calc_accuracy, export_result_dict

from data_preprocessing import get_dataloader
from torch.optim.lr_scheduler import StepLR
from evaluation import collect_samples_indices, save_samples_to_file, confusion_matrix
import torch.nn.functional as F

def main():
    # Path setup
    images_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/lfw2"
    text_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/pairsDevTrain.txt"
    test_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/pairsDevTest.txt"
    exp_number=3
    base_writer_path = f"/sise/home/yuvalgor/deep-learning/Assignment 2/runs/exp{exp_number}"
    optimizers = ["Adam", "RMSprop"]
    dropouts = [0.2, 0]
    batch_norms = [True, False]
    batch_sizes = [8, 32, 128]
    loss_fns_names = ["binary_cross_entropy", "mse_loss"]
    loss_fns = [F.binary_cross_entropy, F.mse_loss]
    use_augmentations = [True, False]
    number_of_models = len(optimizers) * len(dropouts) * len(batch_norms) * len(batch_sizes) * len(loss_fns) * len(use_augmentations)
    print(f"Number of models: {number_of_models}")


    #
    #
    # optimizers = ["Adam"]
    # dropouts = [0.1]
    # batch_norms = [True]
    # batch_sizes = [32]
    # loss_fns = [F.binary_cross_entropy]
    epochs = 10

    results_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_number = 0
    for optimizer_name in optimizers:
        for dropout in dropouts:
            for batch_norm in batch_norms:
                for batch_size in batch_sizes:
                    for loss_fn_name , loss_fn in zip(loss_fns_names, loss_fns):
                        for use_augmentation in use_augmentations:
                            model_number+=1
                            writer_path = f"{base_writer_path}/{optimizer_name}_{dropout}_{batch_norm}_{batch_size}"
                            print(
                                f"|| Running model number {model_number}/{number_of_models}  || \noptimizer: {optimizer_name}, dropout: {dropout}, batch_norm: {batch_norm}, batch_size: {batch_size}, loss_fn: {loss_fn_name}, use_augmentation: {use_augmentation}")
                            train_dataloader, dev_dataloader = get_dataloader(text_path, images_path, batch_size=batch_size,use_augmentation=use_augmentation)
                            test_dataloader = get_dataloader(test_path, images_path, batch_size=batch_size,use_augmentation=False)

                            model = SiameseNetwork(dropout=dropout, batch_norm=batch_norm).to(device)

                            if optimizer_name == "Adam":
                                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                            elif optimizer_name == "SGD":
                                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)
                            else:
                                optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)
                            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                            model, time = train_siamese_network(model, train_dataloader, dev_dataloader, epochs=epochs,loss_fn=loss_fn,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler, device=device, writer_path=writer_path)

                            test_labels, test_predictions = model.get_pred_labels(test_dataloader, device)
                            test_accuracy = calc_accuracy(test_predictions, test_labels)

                            train_labels, train_predictions = model.get_pred_labels(train_dataloader, device)
                            train_accuracy = calc_accuracy(train_predictions, train_labels)

                            val_labels, val_predictions = model.get_pred_labels(dev_dataloader, device)
                            val_accuracy = calc_accuracy(val_predictions, val_labels)

                            print(f"Test accuracy: {test_accuracy:.2f}% time - {time}")
                            results_dict[(optimizer_name, dropout, batch_norm, batch_size,loss_fn_name,use_augmentation)] = {"test_accuracy": test_accuracy,
                                                                                               "train_accuracy": train_accuracy,
                                                                                               "val_accuracy": val_accuracy,
                                                                                               "time": time}
                            # sample_dict = collect_samples_indices(test_dataloader, test_predictions)
                            # save_samples_to_file(sample_dict)
                            # confusion_matrix(sample_dict)

    export_result_dict(results_dict)


if __name__ == "__main__":
    main()
