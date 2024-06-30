import re

import torch
from nltk.tokenize import word_tokenize
import nltk
from torch import nn, optim

nltk.download('punkt')
import pandas as pd

# Special tokens
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
UNK_TOKEN = "<UNK>"



class CustomCrossEntropyLoss(nn.Module):
    def __init__(self,next_line ,ignore_index=0, weight=None):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.next_line=next_line

    def forward(self, predicted, target):
        lens=[]
        target_len = []
        all_lines_count = []
        target_line_count=[]

        for i in range(len(predicted)):
            found_p = False
            found_t = False
            line_count_pred=0
            line_count_target=0
            for j in range(len(predicted[i])):
                if not found_p and predicted[i][j].argmax()==self.next_line:
                    line_count_pred+=1
                if not found_t and target[i][j]==self.next_line:
                    line_count_target+=1
                if not found_p and (predicted[i][j].argmax()==1 or j==len(predicted[i])-1):
                    lens.append(j)
                    found_p=True
                if not found_t and (target[i][j]==1 or j==len(target[i])-1):
                    target_len.append(j)
                    found_t=True
                if found_p and found_t:
                    break

            all_lines_count.append(line_count_pred)
            target_line_count.append(line_count_target)


        alphas = [0.98, 0.01, 0.01]
        predicted=predicted.reshape(-1,predicted.size(-1))
        target=target.view(-1).long()
        loss= self.cross_entropy_loss(predicted, target)
        ## calaculate diif of lens
        len_loss= sum([abs(lens[i]-target_len[i]) for i in range(len(lens))])/len(lens)
        line_loss=sum([abs(all_lines_count[i]-target_line_count[i]) for i in range(len(all_lines_count))])/len(all_lines_count)
        return alphas[0]*loss +alphas[1]*len_loss+alphas[2]*line_loss


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