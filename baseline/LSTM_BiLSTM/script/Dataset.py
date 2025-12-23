from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torchtext.data import get_tokenizer
from TCC.script.textPreproces import preprocess
from TCC.script import config

class PaddedDataset(Dataset):
    def __init__(self, df, vocab, max_length=None):
        self.df = df
        self.vocab = vocab
        self.max_length = max_length
        self.label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.tokenizer = get_tokenizer("basic_english")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['comment_text']
        labels = self.df.iloc[idx][self.label_cols].values.astype(float)

        # Tokenize and numericalize
        tokens = self.tokenizer(preprocess(text))
        if self.max_length:
            tokens = tokens[:self.max_length]
        numericalized = [self.vocab[token] for token in tokens]

        return torch.tensor(numericalized, dtype=torch.long), torch.tensor(labels, dtype=torch.float)


def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    # Filter invalid sequences (length <=0)
    valid_mask = lengths > 0
    if not valid_mask.all():
        texts = [t for t, valid in zip(texts, valid_mask) if valid]
        labels = [l for l, valid in zip(labels, valid_mask) if valid]
        lengths = lengths[valid_mask]

    # Add fallback for empty batch
    if len(texts) == 0:
        return torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 6)), torch.tensor([1])
    # Pad sequences to match longest in batch
    padded_texts = torch.nn.utils.rnn.pad_sequence(
        texts,
        batch_first=True,
        padding_value=config.PAD_IDX
    )

    return padded_texts, torch.stack(labels), lengths