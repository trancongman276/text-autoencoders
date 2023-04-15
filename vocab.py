from collections import Counter
import re

class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
        self.nspecial = 5

    @staticmethod
    def no_accent_vietnamese(s):
        s = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", s)
        s = re.sub(r"[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]", "A", s)
        s = re.sub(r"[èéẹẻẽêềếệểễ]", "e", s)
        s = re.sub(r"[ÈÉẸẺẼÊỀẾỆỂỄ]", "E", s)
        s = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", s)
        s = re.sub(r"[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]", "O", s)
        s = re.sub(r"[ìíịỉĩ]", "i", s)
        s = re.sub(r"[ÌÍỊỈĨ]", "I", s)
        s = re.sub(r"[ùúụủũưừứựửữ]", "u", s)
        s = re.sub(r"[ƯỪỨỰỬỮÙÚỤỦŨ]", "U", s)
        s = re.sub(r"[ỳýỵỷỹ]", "y", s)
        s = re.sub(r"[ỲÝỴỶỸ]", "Y", s)
        s = re.sub(r"[Đ]", "D", s)
        s = re.sub(r"[đ]", "d", s)
        return s

    @staticmethod
    def build(sents, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w for s in sents for w in s]
        no_vnmese = [Vocab.no_accent_vietnamese(w) for w in words]
        words = list(set(words + no_vnmese))
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w', encoding="utf8") as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))
