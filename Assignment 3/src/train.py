import torch
from torch import nn, optim
from tqdm import tqdm
from model import *


def train(model, dataloader, optimizer, criterion, device, epochs, vocabulary, word2vec,teacher_forcing_ratio=0.5):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_idx, (lyrics_features, melody_features, targets) in progress_bar:
            lyrics_features = lyrics_features.to(device)
            melody_features = melody_features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, logits = model(melody_features, lyrics_features, vocabulary, word2vec,teacher_forcing_ratio=teacher_forcing_ratio)
            lyrics = pred_to_lyrics(outputs, vocabulary)
            logits = logits.reshape(-1, logits.size(-1))  # Reshape logits to [batch_size * target_len, vocabulary_size]
            targets = targets.view(-1).long()
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'\nEpoch {epoch + 1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}')

