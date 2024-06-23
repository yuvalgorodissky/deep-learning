import torch
from torch import nn, optim
from tqdm import tqdm
from model import *


def get_generated_lyrics(model, dataloader, device, word2vec ,vocabulary):
    model.eval()
    generated_lyrics = []
    targets = []
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (_, melody_features, target) in progress_bar:
            melody_features = melody_features.to(device)
            predictions = model.predict(melody_features, word2vec, vocabulary, select_strategy='prob', temperature=1.0,
                                        max_length=300)
            lyrics = pred_to_lyrics(predictions, vocabulary)
            generated_lyrics.extend(lyrics)
            targets.extend(pred_to_lyrics(target.int().tolist(), vocabulary))

    return generated_lyrics, targets


