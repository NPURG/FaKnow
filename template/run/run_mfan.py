import json
import pickle
from typing import List

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from template.data.dataset.multi_modal import MultiModalDataset
from template.evaluate.evaluator import Evaluator
from template.model.content_based.multi_modal.mfan import MFAN
from template.train.pgd_trainer import MFANTrainer


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
    padded_tokens = []
    for token_ids in token_ids_list:
        if len(token_ids) >= max_len:
            token_ids = token_ids[:max_len]
        else:
            # 在最前方填充0
            token_ids = [0] * (max_len - len(token_ids)) + token_ids
        padded_tokens.append(token_ids)
    return torch.tensor(padded_tokens)


def load_adj_matrix(path: str, node_num: int):
    with open(path, 'r') as f:
        adj_dict = json.load(f)

    adj_matrix = torch.zeros(size=(node_num, node_num))
    for ori_node, des_nodes in adj_dict.items():
        adj_matrix[int(ori_node), des_nodes] = 1
    return adj_matrix


def run_mfan(path: str, word_vectors: torch.Tensor,
             node_embedding: torch.Tensor, node_num: int,
             adj_matrix: torch.Tensor):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = MultiModalDataset(path, ['text'], tokenize, ['image'], transform)
    size = int(len(dataset) * 0.001)
    train_data, _ = random_split(dataset, [size, len(dataset) - size])
    train_loader = DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True)

    model = MFAN(word_vectors, node_num, node_embedding, adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    evaluator = Evaluator()
    trainer = MFANTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, num_epoch=20)


def main():
    path = "F:\\code\\python\\MFAN\\new_data\\mfan.json"
    pre = "F:\\code\\python\\MFAN\\dataset/weibo/weibo_files"
    adj_path = "F:\\code\\python\\MFAN\\dataset\\weibo\\weibo_files\\original_adj"
    node_num = 6963
    adj_matrix = load_adj_matrix(adj_path, node_num)
    print('loading adj matrix')

    node_embedding = pickle.load(open(pre + "\\node_embedding.pkl", 'rb'))[0]
    _, _, _, word_embeddings, _ = pickle.load(open(pre + "\\train.pkl", 'rb'))
    print('loading embedding')

    run_mfan(path, torch.from_numpy(word_embeddings), torch.from_numpy(node_embedding), node_num,
             adj_matrix)


if __name__ == '__main__':
    main()
