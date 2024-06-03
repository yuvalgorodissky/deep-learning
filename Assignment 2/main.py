import torch
from model import SiameseNetwork
from training import train_siamese_network
from utility import plot_losses,split_dataloader
from data_preprocessing import get_dataloader

def main():
    images_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/lfw2"
    text_path = "/sise/home/yuvalgor/deep-learning/Assignment 2/data/pairsDevTrain.txt"
    batch_size = 32
    train_dataloader, dev_dataloader = split_dataloader(get_dataloader(text_path, images_path, batch_size=batch_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, losses = train_siamese_network(train_dataloader, dev_dataloader, epochs=100, optimizer=optimizer,
                                          model=model, device=device)
    plot_losses(losses, "/sise/home/yuvalgor/deep-learning/Assignment 2/")


if __name__ == "__main__":
    main()