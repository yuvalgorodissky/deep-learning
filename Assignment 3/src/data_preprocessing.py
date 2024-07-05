import mido
import numpy as np
import pretty_midi
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from utils import *
# Set device
from torch.nn.utils.rnn import pad_sequence


def load_midi(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    return midi_data


def load_lyrics(lyrics_path):
    ##read lyrics from csv file, use col name as artist, song, lyrics
    column_names = ['artist', 'song', 'lyrics']
    lyrics = pd.read_csv(lyrics_path, header=None, names=column_names)
    return lyrics


def get_dataset(lyrics_path, midis_path):
    dataset = {
        "artist": [],
        "song": [],
        "lyrics": [],
        "midi_path": []
    }
    lyrics = load_lyrics(lyrics_path)
    for artist, song, lyrics in lyrics.itertuples(index=False):
        artist_template = str(artist).replace(" ", "_")
        if song[0] != " ":
            song = " " + song
        song_template = str(song).replace(" ", "_")

        midi_path = f"{midis_path}/{artist_template}_-{song_template}.mid"

        dataset["artist"].append(artist)
        dataset["song"].append(song)
        dataset["lyrics"].append(lyrics)
        dataset["midi_path"].append(midi_path)

    return Dataset.from_dict(dataset)


def get_features_instrument(midi_file):
    # Get beat times
    beats = midi_file.get_beats()
    beats_length = len(beats)  # Number of columns in the matrix

    # Create a zero matrix of size 128 x beats_length
    all_features = torch.zeros(128, 3, beats_length)

    for instrument in midi_file.instruments:
        program = instrument.program
        for note in instrument.notes:
            pitch = note.pitch
            start = note.start
            end = note.end
            velocity = note.velocity
            duration = end - start

            # Find the closest beat index before the note starts
            closest_beat = max([i for i, beat in enumerate(beats) if beat <= start], default=0)
            # Feature vector
            # Update the matrix
            all_features[program, 0, closest_beat] += duration * pitch
            all_features[program, 1, closest_beat] += duration * velocity
            all_features[program, 2, closest_beat] += duration

    reshaped_features = all_features.view(128 * 3, beats_length)
    return reshaped_features


def get_lyrics_features(lyrics, word2vec_model, vocabulary, word_to_index):
    # Get the word2vec representation of the lyrics
    lyrics_features = []
    words = []
    for word in lyrics.split():
        word = word.lower()
        if word in vocabulary:  # Ensure you handle case sensitivity
            try:
                vector = get_embeddings(word2vec_model, word)
                lyrics_features.append(vector)
                words.append(word_to_index[word])
            except KeyError:
                # This exception handles the case where the word is in the corpus but not in the word2vec model
                lyrics_features.append(get_embeddings(word2vec_model, UNK_TOKEN))
                words.append(word_to_index[UNK_TOKEN])

    lyrics_features.append(get_embeddings(word2vec_model, EOS_TOKEN))
    words.append(word_to_index[EOS_TOKEN])
    lyrics_features_tensor = torch.stack(lyrics_features, dim=0)
    return lyrics_features_tensor, words


def get_features_from_piano_roll(midi_file):
    # Get beat times
    piano_roll= midi_file.get_piano_roll(fs=2)
    piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
    return piano_roll

def get_melody_features(midi_file_path,strategy):
    midi_file = load_midi(midi_file_path)
    x=get_features_instrument(midi_file)
    y=get_features_from_piano_roll(midi_file)
    if strategy == "instrument":
        return  get_features_instrument(midi_file)
    elif strategy == "piano_roll":
        return get_features_from_piano_roll(midi_file)
    else:
        raise ValueError("Invalid strategy")

def collate_fn(batch):
    lyrics_features, melody_features, words = zip(*batch)

    # Determine the maximum lengths
    max_length_lyrics = max([l.shape[0] for l in lyrics_features])
    max_length_melody = max(
        [m.shape[1] for m in melody_features])  # Melody features are assumed to be (features, time_steps)

    # Convert words to tensors and find max length
    words_tensors = [torch.tensor(w) for w in words]  # Convert each list of words into a tensor
    max_length_words = max([w.shape[0] for w in words_tensors])

    # Manually pad lyrics features
    lyrics_features_padded = torch.stack([
        torch.cat([l, torch.zeros(max_length_lyrics - l.shape[0], l.shape[1])]) if l.shape[0] < max_length_lyrics else l
        for l in lyrics_features
    ])

    # Manually pad melody features
    melody_transposed = [m.transpose(0, 1) for m in melody_features]
    melody_padded = torch.stack([
        torch.cat([m, torch.zeros(max_length_melody - m.shape[0], m.shape[1])]) if m.shape[0] < max_length_melody else m
        for m in melody_transposed
    ])

    # Manually pad words
    words_padded = torch.stack([
        torch.cat([w, torch.zeros(max_length_words - w.shape[0])]) if w.shape[0] < max_length_words else w
        for w in words_tensors
    ])

    return lyrics_features_padded, melody_padded, words_padded


def get_dataloader(lyrics_path, midis_path, batch_size, word2vec_model, vocabulary, word_to_index, melody_strategy,shuffle=True):
    dataset = get_dataset(lyrics_path, midis_path)
    new_dataset = []
    for i in range(len(dataset)):
        try:
            lyrics_features, words = get_lyrics_features(dataset["lyrics"][i], word2vec_model, vocabulary, word_to_index)
            melody_features = get_melody_features(dataset["midi_path"][i], melody_strategy)
            new_dataset.append((lyrics_features, melody_features, words))
        except Exception as e:
            print(f"Skipping file {dataset['midi_path'][i]} due to error: {e}")
            continue

    return DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
