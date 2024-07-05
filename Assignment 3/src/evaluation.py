import argparse
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
from utils import *
from generate_lyrics import get_generated_lyrics
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

from collections import Counter
from evaluation_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Train LSTM Seq2Seq model for lyrics generation")
    parser.add_argument('--midi_path', type=str, required=True, help="Path to the MIDI files directory")
    parser.add_argument('--lyrics_path', type=str, required=True, help="Path to the lyrics CSV file")
    parser.add_argument('--test_path', type=str, required=True, help="Path to the test set CSV file")
    parser.add_argument('--models_save_path', type=str, required=True, help="Path to save the trained model")
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
    parser.add_argument('--output_dir', type=str, default="output", help="Output directory")

    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    results = {}
    # get all files from dir
    all_models_files = os.listdir(args.models_save_path)

    predictions_strategies = ["prob", "argmax", "topk"]
    token_kinds = ["first_word", "SOS", "random"]
    for model_path in all_models_files:
        model_melody_strategy = "instrument" if "instrument" in model_path else "piano_roll"
        if model_melody_strategy == "instrument":
            args.input_size_encoder = 128 * 3
        else:
            args.input_size_encoder = 128

        model = load_model_path_eval(args, f'{args.models_save_path}/{model_path}')

        model.to(device)
        test_dataloader = get_dataloader(lyrics_path=args.test_path, midis_path=args.midi_path,
                                         batch_size=args.batch_size,
                                         word2vec_model=model.word2vec, word_to_index=model.word_to_index,
                                         vocabulary=model.vocabulary,
                                         melody_strategy=model_melody_strategy, shuffle=False)
        first_words = [model.vocabulary[x[2][0]] for x in test_dataloader.dataset]
        random_words = random.sample(model.vocabulary, len(test_dataloader.dataset))
        start_tokens = [first_words, [SOS_TOKEN] * len(test_dataloader.dataset), random_words]

        print("*" * 30, " model: ", model_path, "*" * 30)
        for strategy in predictions_strategies:
            for index, start_token in enumerate(start_tokens):
                predictions, targets = get_generated_lyrics(model, test_dataloader, device, start_words=start_token,
                                                            select_strategy=strategy, temperature=1)

                print("Strategy: ", strategy)
                print("Start token: ", start_token)
                jacard_results_n_1 = compare_songs(predictions, targets, 1)
                jacard_results_n_2 = compare_songs(predictions, targets, 2)
                for song_idx, (jacard_result_n_1, jacard_result_n_2) in enumerate(zip(jacard_results_n_1, jacard_results_n_2)):
                    results[(song_idx,model_path, strategy, token_kinds[index])] = (
                    predictions[song_idx], targets[song_idx], jacard_result_n_1, jacard_result_n_2)


    for song_idx, model_path, strategy, token_kind in results.keys():
        predictions, targets, jacard_result_n_1, jacard_result_n_2 = results[(song_idx, model_path, strategy, token_kind)]
        with open(f'{args.output_dir}/song-{song_idx}.txt', "a") as f:
            f.write("Strategy: " + strategy + "\n")
            f.write("Start token: " + token_kind + "\n")
            f.write("Prediction: " + predictions + "\n")
            f.write("Target: " + targets + "\n")
            f.write("Jacard n=1: " + str(jacard_result_n_1) + "\n")
            f.write("Jacard n=2: " + str(jacard_result_n_2) + "\n")
            f.write("\n")
            f.write("*" * 50)
            f.write("\n")

    #save results as csv file
    export_result_dict(results,args.output_dir)



if __name__ == "__main__":
    main()
