import torch
import torch.nn as nn


class bilstm(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout =  nn.Dropout(p = 0.3)
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted = False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc1(hidden_output)
        out = self.dropout(out)
        return torch.sigmoid(out)