import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
# Assuming it's the correct DataLoader from previous context
from train import train
from data_preprocessing import get_dataloader
import gensim
import gensim.downloader as api
import nltk
from nltk.corpus import words
from utils import extract_language, save_model, load_model, pretty_print_params, CustomCrossEntropyLoss,set_seed
from generate_lyrics import get_generated_lyrics
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from model import lstm_seq2seq



def get_args():
    parser = argparse.ArgumentParser(description="Train LSTM Seq2Seq model for lyrics generation")
    parser.add_argument('--midi_path', type=str, required=True, help="Path to the MIDI files directory")
    parser.add_argument('--lyrics_path', type=str, required=True, help="Path to the lyrics CSV file")
    parser.add_argument('--test_path', type=str, required=True, help="Path to the test set CSV file")
    parser.add_argument('--model_save_path', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--writer_path', type=str, required=True, help="Path for TensorBoard logs")
    parser.add_argument('--input_size_encoder', type=int, default=128, help="Input size for the encoder")
    parser.add_argument('--hidden_size_encoder', type=int, default=256, help="Hidden size for the encoder")
    parser.add_argument('--input_size_decoder', type=int, default=300, help="Input size for the decoder")
    parser.add_argument('--hidden_size_decoder', type=int, default=512, help="Hidden size for the decoder")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the LSTM")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--melody_strategy', type=str, default="piano_roll", help="Melody strategy")
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help="Teacher forcing ratio")

    return parser.parse_args()


def main():
    args = get_args()
    if args.melody_strategy == "instrument":
        args.input_size_encoder = 128 * 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2vec_model = api.load('word2vec-google-news-300')
    vocabulary, word_to_index = extract_language(args.lyrics_path, word2vec_model)
    set_seed(42)
    writer = SummaryWriter(args.writer_path)  # Initialize TensorBoard writer

    dataloader = get_dataloader(lyrics_path=args.lyrics_path, midis_path=args.midi_path, batch_size=args.batch_size,
                                word2vec_model=word2vec_model, word_to_index=word_to_index, vocabulary=vocabulary,
                                melody_strategy=args.melody_strategy)

    # Initialize model
    model = lstm_seq2seq(
        input_size_encoder=args.input_size_encoder,
        hidden_size_encoder=args.hidden_size_encoder,
        input_size_decoder=args.input_size_decoder,
        hidden_size_decoder=args.hidden_size_decoder,
        vect_size_decoder=len(vocabulary),
        num_layers=args.num_layers,
        word2vec_model=word2vec_model,
        word_to_index=word_to_index,
        vocabulary=vocabulary
    )

    pretty_print_params(model)
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CustomCrossEntropyLoss(next_line=word_to_index['&'], ignore_index=0)

    # Training
    train(model, dataloader, optimizer, criterion, device, args.epochs, teacher_forcing_ratio=args.teacher_forcing,
          writer=writer)

    save_model(model, args.model_save_path)




if __name__ == "__main__":
    main()
