import json
import pickle
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split

from template.data.dataset.mfan_dataset import JsonDataset
from template.evaluate.evaluator import Evaluator
from template.model.multi_modal.mfan import MFAN
from template.train.trainer import BaseTrainer


def tokenize(texts: List[str]):
    with open("F:\\code\\python\\MFAN\\dataset\\weibo\\weibo_files\\vocab.pkl",
              'rb') as f:
        vocab = pickle.load(f)
    # 采用分字而非分词
    token_ids_list = [[vocab[word] for word in text if word in vocab]
                      for text in texts]
    token_ids_list = pad_sequence(token_ids_list, max_len=50)
    return token_ids_list


def pad_sequence(token_ids_list: List[List[int]], max_len=50):
    paded_tokens = []
    for token_ids in token_ids_list:
        if len(token_ids) >= max_len:
            token_ids = token_ids[:max_len]
        else:
            # 在最前方填充0
            token_ids = [0] * (max_len - len(token_ids)) + token_ids
        paded_tokens.append(token_ids)
    return torch.tensor(paded_tokens)


def transform(path: str) -> torch.Tensor:
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return trans(Image.open(path).convert("RGB"))


def load_adj_matrix(path: str, node_num: int):
    with open(path, 'r') as f:
        adj_dict = json.load(f)

    adj_matrix = torch.zeros(size=(node_num, node_num))
    for ori_node, des_nodes in adj_dict.items():
        adj_matrix[int(ori_node), des_nodes] = 1
    return adj_matrix


def run_mfan(path: str, word_vectors: np.ndarray, max_len: int,
             node_embedding: np.ndarray, node_num: int,
             adj_matrix: torch.Tensor):
    dataset = JsonDataset(path, transform, tokenize)
    size = int(len(dataset) * 0.2)
    train_data, _ = random_split(dataset, [size, len(dataset) - size])

    model = MFAN(word_vectors, max_len, node_num, node_embedding, adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    evaluator = Evaluator()
    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_data,
                batch_size=64,
                epochs=20,
                validate_size=0.2,
                saved=False)


if __name__ == '__main__':
    path = "F:\\code\\python\\MFAN\\test\\weibo.json"
    pre = "F:\\code\\python\\MFAN\\dataset/weibo/weibo_files"
    adj_path = "F:\\code\\python\\MFAN\\dataset\\weibo\\weibo_files\\original_adj"
    max_len = 50
    node_num = 6963
    adj_matrix = load_adj_matrix(adj_path, node_num)
    print('loading adj matrix')

    node_embedding = pickle.load(open(pre + "\\node_embedding.pkl", 'rb'))[0]
    _, _, _, word_embeddings, _ = pickle.load(open(pre + "\\train.pkl", 'rb'))
    print('loading embedding')

    run_mfan(path, word_embeddings, max_len, node_embedding, node_num,
             adj_matrix)
