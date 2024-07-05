import torch
from torch import nn, optim
from tqdm import tqdm
from model import *
from utils import *

def get_generated_lyrics(model, dataloader,device, start_words,select_strategy='prob', temperature=1.0,max_length=300,):
    model.eval()
    generated_lyrics = []
    targets = []
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for batch_idx, (_, melody_features, target) in progress_bar:
            melody_features = melody_features.to(device)
            predictions = model.predict(melody_features, select_strategy=select_strategy, temperature=temperature,
                                        max_length=max_length,start_word=start_words[batch_idx])
            if start_words[batch_idx]!='<SOS>':
                lyrics = [f"{start_words[batch_idx]} {pred_to_lyrics(predictions, model.vocabulary)[0]}"]
            else:
                lyrics = pred_to_lyrics(predictions, model.vocabulary)
            generated_lyrics.extend(lyrics)
            targets.extend(pred_to_lyrics(target.int().tolist(), model.vocabulary))

    return generated_lyrics, targets


