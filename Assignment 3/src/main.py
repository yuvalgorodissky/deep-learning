import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Assuming it's the correct DataLoader from previous context
from train import train
from model import lstm_seq2seq  # Assuming your model's module is correctly named
from data_preprocessing import get_dataloader
import gensim
import gensim.downloader as api
import nltk
from nltk.corpus import words
from utils import extract_language, save_model, load_model, pretty_print_params
from generate_lyrics import get_generated_lyrics


# Set device

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midi_path = "/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/midi_files"
    lyrics_path = "/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/lyrics_train_set1.csv"
    word2vec_model = api.load('word2vec-google-news-300')
    test_path = "/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/lyrics_test_set.csv"
    model_save_path = "/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/models/seq2seq_model_0.5.pth"
    vocabulary, word_to_index = extract_language(lyrics_path, word2vec_model)

    teacher_forcing = 0.5
    # Hyperparameters
    input_size_encoder = 128 * 3  # Placeholder, adjust according to your actual input size
    hidden_size_encoder = 512
    input_size_decoder = 300
    hidden_size_decoder = 512
    vect_size_decoder = len(vocabulary)
    num_layers = 2
    learning_rate = 0.001
    batch_size = 4
    epochs = 150

    dataloader = get_dataloader(lyrics_path=lyrics_path, midis_path=midi_path, batch_size=batch_size,
                                word2vec_model=word2vec_model, word_to_index=word_to_index, vocabulary=vocabulary)

    # Initialize model
    model = lstm_seq2seq(
        input_size_encoder=input_size_encoder,
        hidden_size_encoder=hidden_size_encoder,
        input_size_decoder=input_size_decoder,
        hidden_size_decoder=hidden_size_decoder,
        vect_size_decoder=vect_size_decoder,
        num_layers=num_layers
    )

    pretty_print_params(model)
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Adjust according to your output and targets

    # Training
    train(model, dataloader, optimizer, criterion, device, epochs,vocabulary,word2vec_model,teacher_forcing_ratio=teacher_forcing)

    # Save model
    save_model(model, model_save_path)

    test_dataloader = get_dataloader(lyrics_path=test_path, midis_path=midi_path, batch_size=batch_size,
                                     word2vec_model=word2vec_model, word_to_index=word_to_index, vocabulary=vocabulary)

    # Load model
    model = load_model(model_save_path)
    model.to(device)
    # Test
    predictions, targets = get_generated_lyrics(model, test_dataloader, device, word2vec_model, vocabulary)

    for i in range(5):
        print("prediction: ", predictions[i])
        print("target: ", targets[i])


if __name__ == "__main__":
    main()
