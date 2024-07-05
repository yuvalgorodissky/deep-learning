from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from model import *

def train(model, dataloader, optimizer, criterion, device, epochs, writer,teacher_forcing_ratio=0.5):
    model.train()
    model.to(device)
    scaler = GradScaler()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_idx, (lyrics_features, melody_features, targets) in progress_bar:
            lyrics_features = lyrics_features.to(device)
            melody_features = melody_features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs, logits = model(melody_features, lyrics_features, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            writer.add_scalar('Training loss', loss.item(), epoch * len(dataloader) + batch_idx)

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'\nEpoch {epoch + 1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}')


