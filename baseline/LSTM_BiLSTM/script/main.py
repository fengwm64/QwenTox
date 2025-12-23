import nltk
import torch
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from TCC.script.Dataset import PaddedDataset, collate_batch
from TCC.script.loadData import loadCSV
from TCC.script.train import train
from TCC.script.untils import yield_tokens
from TCC.script import config

BATCH_SIZE = 64
MAX_SEQ_LEN = 256

train_data, test_data, test_labels_data, sample_data = loadCSV()
comments = list(train_data["comment_text"])
train_iter = iter(comments)
tokenizer = get_tokenizer("basic_english")
# Build vocabulary with size limit
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer),
    specials=["<pad>", "<unk>"],
    max_tokens=30002  # 30K + 2 special tokens for unkown tokens and padding
)
vocab.set_default_index(vocab["<unk>"])
PAD_IDX = vocab['<pad>']
config.PAD_IDX = PAD_IDX

def ensure_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def main(fileName):
    print(f"Final vocabulary size: {len(vocab)}")
    train_df, val_df = train_test_split(train_data.iloc[:, 1:], test_size=0.2)
    train_dataset = PaddedDataset(train_df, vocab, max_length=256)
    val_dataset = PaddedDataset(val_df, vocab, max_length=256)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    train(train_loader, val_loader, 5, 0.001, fileName, vocab, PAD_IDX)


if __name__ == "__main__":
    ensure_nltk_data()
    main("lstm")