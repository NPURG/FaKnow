import json
import pickle
from typing import List, Dict

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.mfan import MFAN
from faknow.train.pgd_trainer import MFANTrainer
from faknow.utils.util import dict2str


class MFANTokenizer:
    def __init__(self, vocab: Dict[str, int], max_len=50) -> None:
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, texts: List[str]) -> Tensor:
        token_ids = []
        for text in texts:

            token_id = [self.vocab[word] for word in text]
            real_len = len(token_id)
            if real_len < self.max_len:
                # padding zero in the front
                token_id = [0] * (self.max_len - real_len) + token_id
            elif real_len > self.max_len:
                token_id = token_id[:self.max_len]
            token_ids.append(token_id)

        return torch.tensor(token_ids)


def transform(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return trans(img)


def load_adj_matrix(path: str, node_num: int):
    with open(path, 'r') as f:
        adj_dict = json.load(f)

    adj_matrix = torch.zeros(size=(node_num, node_num))
    for ori_node, des_nodes in adj_dict.items():
        adj_matrix[int(ori_node), des_nodes] = 1
    return adj_matrix


def run_mfan(train_path: str,
             node_embedding: torch.Tensor,
             node_num: int,
             adj_matrix: torch.Tensor,
             vocab: Dict[str, int],
             word_vectors: torch.Tensor,
             max_len=50,
             batch_size=64,
             num_epochs=20,
             lr=2e-3,
             metrics: List = None,
             validate_path: str = None,
             test_path: str = None
             ):
    tokenize = MFANTokenizer(vocab, max_len)

    train_data = MultiModalDataset(train_path, ['text'], tokenize, ['image'], transform)
    train_loader = DataLoader(train_data, batch_size, True)

    if validate_path:
        validate_data = MultiModalDataset(validate_path, ['text'], tokenize, ['image'], transform)
        validate_loader = DataLoader(validate_data, batch_size, True)
    else:
        validate_loader = None

    model = MFAN(word_vectors, node_num, node_embedding, adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    evaluator = Evaluator(metrics)
    trainer = MFANTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, num_epochs, validate_loader)

    if test_path is not None:
        test_data = MultiModalDataset(test_path, ['text'], tokenize, ['image'], transform)
        test_loader = DataLoader(test_data, batch_size, True)
        test_result = trainer.evaluate(test_loader)
        print('test result: ', dict2str(test_result))


def main():
    path = "F:\\code\\python\\MFAN\\new_data\\mfan.json"
    pre = "F:\\code\\python\\MFAN\\dataset/weibo/weibo_files"

    adj_path = "F:\\code\\python\\MFAN\\dataset\\weibo\\weibo_files\\original_adj"
    node_num = 6963
    adj_matrix = load_adj_matrix(adj_path, node_num)
    node_embedding = pickle.load(open(pre + "\\node_embedding.pkl", 'rb'))[0]
    print('loading adj matrix')

    _, _, _, word_embeddings, _ = pickle.load(open(pre + "\\train.pkl", 'rb'))
    vocab = pickle.load(open(pre + "\\vocab.pkl", 'rb'))
    print('loading embedding')

    run_mfan(path, torch.from_numpy(node_embedding), node_num, adj_matrix,
             vocab, torch.from_numpy(word_embeddings))


if __name__ == '__main__':
    main()
