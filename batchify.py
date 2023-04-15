import numpy as np
import torch

from vocab import Vocab

THRESH_NOPUNCT = 0.7


def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    keep = np.random.rand(len(x)) < THRESH_NOPUNCT
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        s_idx_label = []
        for w in s:
            if w in vocab.word2idx:
                if keep[0]:
                    s_idx_label.append(vocab.word2idx[Vocab.no_accent_vietnamese(w)])
                else:
                    s_idx_label.append(vocab.word2idx[w])
            else:
                s_idx_label.append(vocab.unk)
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx_label + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), torch.LongTensor(
        x_eos
    ).t().contiguous().to(
        device
    )  # time * batch


def get_batches(data, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i + batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch(data[i:j], vocab, device))
        i = j
    return batches, order
