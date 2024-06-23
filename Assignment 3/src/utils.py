import re

import torch
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
import pandas as pd

# Special tokens
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
UNK_TOKEN = "<UNK>"



def pretty_print_params(model):
    print("{:<30} {:>15}".format("Layer", "Num Parameters"))
    print("-" * 45)
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()  # total number of elements in the parameter
            print(f"{name:<30} {param_count:>15,}")  # format with commas
            total_params += param_count
    print("-" * 45)
    print(f"{'Total':<30} {total_params:>15,}")


def pred_to_lyrics(predictions,vocabulary):
    lyrics = []
    for prediction in predictions:
        lyric = []
        for word in prediction:
            if word == '&':
                word='\n'
            lyric.append(vocabulary[word])
            if word == EOS_TOKEN:
                break   # Stop generating lyrics if the model predicts the end of the song
        lyrics.append(' '.join(lyric))
    return lyrics


def get_one_hot_torch_vector(index, size):
    vector = torch.zeros(size)
    vector[index] = 1
    return vector


def extract_language(path,word2vec):
    # Return a set of all words within the entire songs and a dictionary mapping words to their indices
    vocabulary = set()
    df = pd.read_csv(path, header=None)
    lyrics = df[2].values  # Directly access the lyrics column
    for lyric in lyrics:
        words = word_tokenize(lyric)
        for word in words:
            if word in word2vec:
                vocabulary.add(word)

    vocabulary = list(vocabulary)
    vocabulary = [PAD_TOKEN, EOS_TOKEN, SOS_TOKEN, UNK_TOKEN] + vocabulary

    # Create a dictionary that maps each word to its index
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

    return vocabulary, word_to_index


def get_embeddings(word2vec, word):
    if word == PAD_TOKEN:
        return get_one_hot_torch_vector(0, 300)
    if word == EOS_TOKEN:
        return get_one_hot_torch_vector(1, 300)
    if word == SOS_TOKEN:
        return get_one_hot_torch_vector(2, 300)
    if word == UNK_TOKEN:
        return get_one_hot_torch_vector(3, 300)
    return torch.tensor(word2vec[word])


def save_model(model, path):
   # Save the model to the specified path
   torch.save(model, path)
   print("Model saved at", path)

def load_model(path):
    # Load the model from the specified path
    model = torch.load(path)
    print("Model loaded from", path)
    return model