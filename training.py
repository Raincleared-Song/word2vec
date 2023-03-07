import os
import scipy
import torch
import random
import shutil
import numpy as np
from tqdm import tqdm
from config import Config
from dataset import Corpus
from model import SkipGram
from utils import save_json
import torch.optim as optim
from torch.utils.data import DataLoader


global_loader_generator = torch.Generator()


def init_all():
    os.makedirs(Config.output_path, exist_ok=True)
    shutil.copy('config.py', os.path.join(Config.output_path, 'config.py'))
    global global_loader_generator
    seed = Config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    global_loader_generator.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    determine = torch.use_deterministic_algorithms if 'use_deterministic_algorithms' in dir(torch) \
        else torch.set_deterministic
    determine(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def evaluate(vocab2id: dict, embedding):
    # glove.6B.100d 0.5483502272978049
    # glove.6B.200d 0.5777980450176106
    # glove.6B.300d 0.6040760924030992
    benchmark = 'data/wordsim353/combined.csv'
    with open(benchmark) as fin:
        lines = fin.readlines()
    word_a_ids, word_b_ids, scores = [], [], []
    for line in lines[1:]:
        word_a, word_b, score = line.strip().split(',')
        word_a, word_b = word_a.lower(), word_b.lower()
        word_a_ids.append(vocab2id[word_a])
        word_b_ids.append(vocab2id[word_b])
        scores.append(float(score))
    word_a_ids, word_b_ids, scores = np.array(word_a_ids), np.array(word_b_ids), np.array(scores)
    word_a_embedding = embedding[word_a_ids, :]
    word_b_embedding = embedding[word_b_ids, :]
    pred_scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            for v1, v2 in zip(word_a_embedding, word_b_embedding)])
    return scipy.stats.pearsonr(pred_scores, scores).statistic


def train():
    corpus = Corpus(Config.input_path)
    data_loader = DataLoader(corpus, batch_size=Config.batch_size, shuffle=True,
                             num_workers=4, collate_fn=Corpus.collate_fn)
    vocab_size = len(corpus.id2vocab)
    model = SkipGram(vocab_size, Config.embed_dim)
    model = model.to(Config.device)
    os.makedirs(Config.output_path, exist_ok=True)
    save_json(corpus.id2vocab, os.path.join(Config.output_path, 'id_to_vocab.json'))
    save_json(corpus.vocab2id, os.path.join(Config.output_path, 'vocab_to_id.json'))
    optim_cls = optim.SparseAdam if Config.sparse else optim.Adam
    zs_res = evaluate(corpus.vocab2id, model.embedding.weight.cpu().detach().numpy())
    print('zero shot score:', zs_res)

    ave_losses, correlations = [], []
    max_epoch, max_res = -1, 0.
    for epoch in range(Config.epoch_number):
        print('training epoch:', epoch)
        optimizer = optim_cls(model.parameters(), lr=Config.init_learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_loader))
        epoch_loss = 0.

        for uids, vids, neg_vids in tqdm(data_loader, desc=f'epoch {epoch}'):
            uids = uids.to(Config.device)
            vids = vids.to(Config.device)
            neg_vids = neg_vids.to(Config.device)

            optimizer.zero_grad()
            loss = model(uids, vids, neg_vids)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        ave_loss = epoch_loss / len(data_loader)
        model.save_model(os.path.join(Config.output_path, f'embed_{epoch}.npy'))
        ave_losses.append(ave_loss)
        save_json(ave_losses, os.path.join(Config.output_path, 'ave_losses.json'))
        eval_score = evaluate(corpus.vocab2id, model.embedding.weight.cpu().detach().numpy())
        correlations.append(eval_score)
        if eval_score > max_res:
            max_epoch, max_res = epoch, eval_score
        save_json(correlations, os.path.join(Config.output_path, 'eval_results.json'))

        print(f'finished epoch {epoch}: loss {ave_loss}, result: {eval_score}')

    print('max epoch:', max_epoch, 'max result:', max_res)
    if Config.epoch_number >= 5:
        print('epoch 5 result:', correlations[4])
    if Config.epoch_number >= 10:
        print('epoch 10 result:', correlations[9])
