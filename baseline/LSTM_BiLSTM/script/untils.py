import torch
from TCC.script.textPreproces import preprocess

def yield_tokens(data_iter,tokenizer):
    for text in data_iter:
        cleaned_text = preprocess(text)
        tokens = [
            token for token in tokenizer(cleaned_text)
            if 1 < len(token) < 25
        ]
        yield tokens


def text_pipeline(text, vocab, tokenizer):
    return [
        vocab[token] if token in vocab else vocab['<unk>']
        for token in tokenizer(text)
    ]

def label_pipeline(labels):
    return torch.FloatTensor(labels)