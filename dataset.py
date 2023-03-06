import torch
import numpy as np
from config import Config
from torch.utils.data import Dataset


class Corpus(Dataset):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.trimmed_word_ids = []  # word ids after discarding some most common words
        self.vocab2id, self.id2vocab = {}, []
        self.vocab2freq = {}
        self.total_word_count = 0
        # negative sampling probability and 1 - discard probability
        self.id2neg, self.vocab2dis = [], {}
        self.neg_word_ls, self.neg_ptr = [], 0

        self.initialize()

    def initialize(self):
        with open(self.input_file) as fin:
            text = fin.read().strip()
        tokens = text.split()
        for token in tokens:
            if token not in self.vocab2id:
                self.vocab2id[token] = len(self.id2vocab)
                self.id2vocab.append(token)
                self.vocab2freq[token] = 0
                self.id2neg.append(0.)
            self.vocab2freq[token] += 1
            self.total_word_count += 1
        assert len(self.vocab2id) == len(self.id2vocab) == len(self.vocab2freq)
        print('Size of the vocabulary:', len(self.id2vocab))  # 253854

        frequencies = np.array(list(self.vocab2freq.values())) / self.total_word_count
        norm = np.sum(frequencies ** Config.neg_pow)
        for token, freq in self.vocab2freq.items():
            freq /= self.total_word_count
            self.id2neg[self.vocab2id[token]] = freq ** Config.neg_pow / norm
            self.vocab2dis[token] = np.sqrt(Config.discard_t / freq) + Config.discard_t / freq

        neg_count = np.round(np.array(self.id2neg) * Config.neg_table_size)
        for wid, cnt in enumerate(neg_count):
            self.neg_word_ls += [wid] * int(cnt)
        self.neg_word_ls = np.array(self.neg_word_ls)
        np.random.shuffle(self.neg_word_ls)

        self.trimmed_word_ids = [self.vocab2id[word] for word in tokens if np.random.random() < self.vocab2dis[word]]
        print(len(tokens), len(self.trimmed_word_ids))  # 17005207 9695403

    def negative_sampling(self, target: int, size: int):
        sampled_wids = []
        while len(sampled_wids) < size:
            if self.neg_ptr >= len(self.neg_word_ls):
                np.random.shuffle(self.neg_word_ls)
                self.neg_ptr = 0
            next_id = self.neg_word_ls[self.neg_ptr]
            self.neg_ptr += 1
            if next_id != target:
                sampled_wids.append(next_id)
        # print('sampled:', size, 'ptr:', self.neg_ptr)
        return sampled_wids

    def __len__(self):
        return len(self.trimmed_word_ids)

    def __getitem__(self, item):
        target_id, word_ids = self.trimmed_word_ids[item], self.trimmed_word_ids
        batch = []
        for wid in word_ids[max(0, item - Config.window_size):(item + Config.window_size)]:
            if wid != target_id:
                batch.append((target_id, wid, self.negative_sampling(target_id, Config.neg_count)))
        return batch

    @staticmethod
    def collate_fn(batches):
        uids, vids, neg_vids = [], [], []
        for batch in batches:
            if len(batch) == 0:
                continue
            for uid, vid, neg_vid in batch:
                uids.append(uid)
                vids.append(vid)
                neg_vids.append(neg_vid)
        return torch.LongTensor(uids), torch.LongTensor(vids), torch.LongTensor(neg_vids)
