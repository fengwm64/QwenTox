import torch
import torch.nn as nn


class lstm(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 添加Sigmoid

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),  # 确保lengths在CPU上
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        out = self.fc1(hidden[-1])
        out = self.sigmoid(out)
        return out