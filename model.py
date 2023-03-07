import torch
import numpy as np
import torch.nn as nn
from config import Config
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=Config.sparse)
        self.v_embedding = nn.Embedding(vocab_size, embed_dim, sparse=Config.sparse)
        init_range = 1.0 / self.embed_dim
        nn.init.uniform_(self.embedding.weight.data, -init_range, init_range)
        nn.init.constant_(self.v_embedding.weight.data, 0)

    def forward(self, wids, vids, neg_vids):
        w_embedding = self.embedding(wids)
        v_embedding = self.v_embedding(vids)
        neg_v_embedding = self.v_embedding(neg_vids)

        # context word prediction
        context_score = torch.sum(torch.mul(w_embedding, v_embedding), dim=1)
        context_score = torch.clamp(context_score, min=-10, max=10)
        context_score = - F.logsigmoid(context_score)

        # negative sampling
        neg_score = torch.bmm(neg_v_embedding, w_embedding.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, min=-10, max=10)
        neg_score = - torch.sum(F.logsigmoid(- neg_score), dim=1)

        loss = torch.mean(context_score + neg_score)
        return loss

    def save_model(self, path: str):
        embeddings = self.embedding.weight.cpu().detach().numpy()
        np.save(path, embeddings)
