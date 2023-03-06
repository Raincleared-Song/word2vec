import json
import numpy as np


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def complex_sample(samples, size=None, replace=True, p=None):
    sampled_idx = np.random.choice(len(samples), size=size, replace=replace, p=p)
    if size is None:
        return samples[sampled_idx]
    res = []
    for idx in sampled_idx:
        res.append(samples[idx])
    return res
