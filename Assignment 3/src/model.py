import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import *


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, decoder hidden dim]
        # encoder_outputs: [batch size, src len, encoder hidden dim]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Compute energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)

        # Compute attention
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence with dropout '''

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.5):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True,bidirectional=True)


    def forward(self, melody_input):
        outputs, (hidden, cell) = self.lstm(melody_input)
        # Concatenate the hidden states from both directions
        hidden = self._cat_directions(hidden)
        cell = self._cat_directions(cell)
        return outputs, hidden, cell

    def _cat_directions(self, h):
        ''' Concatenate the hidden states from both directions '''
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=2)


    # def init_hidden(self, batch_size):
    #     return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
    #             torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder with dropout '''

    def __init__(self, input_size, hidden_size,encoder_hidden_size, num_layers=1, dropout=0.5):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size*2
        self.attention = Attention(self.hidden_size ,self.hidden_size )  # Account for bidirectional encoder
        self.lstm = nn.LSTM(input_size=input_size+self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, hidden, cell, encoder_outputs):
        # Calculate attention weights from the last hidden state of the decoder
        attention_weights = self.attention(hidden[-1], encoder_outputs)

        # Apply attention weights to encoder outputs to get context
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)

        # Ensure `x` is [batch size, 1, decoder input features]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Combine embedded input word and context
        lstm_input = torch.cat((x.squeeze(1), context), dim=1).unsqueeze(1)

        # Pass the combined input through the LSTM
        out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        return out, hidden, cell, attention_weights


class lstm_seq2seq(nn.Module):
    ''' Train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size_encoder, hidden_size_encoder, input_size_decoder, hidden_size_decoder,
                 vect_size_decoder, num_layers=1, dropout=0.3):
        super(lstm_seq2seq, self).__init__()
        self.encoder = lstm_encoder(input_size=input_size_encoder, hidden_size=hidden_size_encoder,
                                    num_layers=num_layers)
        self.decoder = lstm_decoder(input_size=input_size_decoder, hidden_size=hidden_size_decoder,
                                    encoder_hidden_size=hidden_size_encoder,num_layers=num_layers)

        self.linear = nn.Linear(self.decoder.hidden_size, vect_size_decoder)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(vect_size_decoder)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Weights of the input-hidden layers
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Weights of the hidden-hidden layers
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # Biases
                param.data.fill_(0)
            elif 'linear' in name:  # Linear layer weights
                torch.nn.init.kaiming_normal_(param.data)

    def forward(self, melody_input, lyrics_input, vocabulary, word2vec, teacher_forcing_ratio=1,
                select_strategy='argmax',
                temperature=1.0):
        batch_size = lyrics_input.shape[0]
        target_length = lyrics_input.shape[1]
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(melody_input)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        predictions = []
        logits = []
        last_output=lyrics_input[:, 0, :].view(batch_size, 1, self.decoder.input_size)
        ##expend to batch size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for t in range(target_length):
            last_output = last_output.to(device)
            decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(last_output, decoder_hidden, decoder_cell,encoder_outputs)
            last_output.detach()
            decoder_output = self.dropout(self.linear(decoder_output))
            decoder_output = self.layer_norm(decoder_output)
            # Apply temperature to logits before softmax
            decoder_output = decoder_output / temperature
            logits.append(decoder_output)
            decoder_output = torch.nn.functional.softmax(decoder_output, dim=2)

            if select_strategy == 'prob':
                decoder_output = torch.multinomial(decoder_output.squeeze(1), 1).squeeze(1)
            elif select_strategy == 'argmax':
                decoder_output = torch.argmax(decoder_output, dim=2).squeeze(
                    1)  # Changed dim to 2 because you're squeezing dim 1 later.
            elif select_strategy == 'topk':
                topk_probs, topk_indices = torch.topk(decoder_output, k=5, dim=2)
                topk_probs = topk_probs.squeeze(1)
                decoder_output = torch.multinomial(topk_probs, 1).squeeze(1)
            else:
                raise ValueError('Invalid select_strategy')
            predictions.append(decoder_output)

            decoder_output = [vocabulary[idx] for idx in decoder_output.tolist()]

            # Implementing teacher forcing
            if torch.rand(1).item() <= teacher_forcing_ratio:
                last_output = lyrics_input[:, t, :].view(batch_size, 1, self.decoder.input_size)
            else:
                words_embeddings = []
                for i in range(batch_size):
                    words_embeddings.append(get_embeddings(word2vec, decoder_output[i]))
                last_output = torch.stack(words_embeddings, dim=0).view(batch_size, 1, self.decoder.input_size)

        predictions = torch.stack(predictions, dim=0)
        logits = torch.stack(logits, dim=0)

        return predictions.T, logits.squeeze(2).transpose(0, 1)

    def predict(self, melody_input, word2vec, vocabulary, select_strategy='argmax', temperature=1.0, max_length=300):
        batch_size = melody_input.shape[0]
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(melody_input)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        predictions = []
        last_output = get_embeddings(word2vec, SOS_TOKEN)
        last_output = last_output.expand(batch_size, 1, self.decoder.input_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for t in range(max_length):
            last_output = last_output.to(device)
            decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(last_output, decoder_hidden, decoder_cell,encoder_outputs)
            last_output.detach()
            decoder_output = self.linear(decoder_output)
            decoder_output = self.layer_norm(decoder_output)
            # Apply temperature to logits before softmax
            decoder_output = decoder_output / temperature
            decoder_output = torch.nn.functional.softmax(decoder_output, dim=2)
            if select_strategy == 'prob':
                decoder_output = torch.multinomial(decoder_output.squeeze(1), 1).squeeze(1)
            elif select_strategy == 'argmax':
                decoder_output = torch.argmax(decoder_output, dim=2).squeeze(
                    1)
            elif select_strategy == 'topk':
                topk_probs, topk_indices = torch.topk(decoder_output, k=5, dim=2)
                topk_probs = topk_probs.squeeze(1)
                decoder_output = torch.multinomial(topk_probs, 1).squeeze(1)
            else:
                raise ValueError('Invalid select_strategy')
            predictions.append(decoder_output)
            words_embeddings = []
            decoder_output = [vocabulary[idx] for idx in decoder_output.tolist()]
            for i in range(batch_size):
                words_embeddings.append(get_embeddings(word2vec, decoder_output[i]))
            last_output = torch.stack(words_embeddings, dim=0).view(batch_size, 1, self.decoder.input_size)

        predictions = torch.stack(predictions, dim=0)
        return predictions.T




