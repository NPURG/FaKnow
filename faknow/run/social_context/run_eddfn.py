import pickle
from typing import List, Dict, Any

import torch
import torch.optim
import yaml
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.eddfn import EDDFN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import lsh_data_selection

__all__ = ['run_eddfn', 'run_eddfn_from_yaml']


def run_eddfn(train_pool_input: Tensor,
              train_pool_label: Tensor,
              domain_embedding: Tensor,
              budget_size=0.8,
              num_h=10,
              batch_size=32,
              num_epochs=100,
              lr=0.02,
              metrics: List = None,
              device='cpu'):
    input_size = train_pool_input.shape[-1]
    domain_size = domain_embedding.shape[-1]

    train_pool_set = TensorDataset(train_pool_input, domain_embedding, train_pool_label)

    selected_ids = lsh_data_selection(domain_embedding,
                                      int(len(train_pool_set) * budget_size),
                                      num_h)
    train_set = TensorDataset(*train_pool_set[selected_ids])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = EDDFN(input_size, domain_size)
    evaluator = Evaluator(metrics)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = BaseTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, num_epochs=num_epochs)


def _parse_kargs(config: Dict[str, Any]) -> Dict[str, Any]:
    with open(config['train_pool_input'], 'rb') as f:
        config['train_pool_input'] = pickle.load(f)
    with open(config['train_pool_label'], 'rb') as f:
        config['train_pool_label'] = pickle.load(f)
    with open(config['domain_embedding'], 'rb') as f:
        config['domain_embedding'] = pickle.load(f)
    return config


def run_eddfn_from_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_eddfn(**_parse_kargs(_config))
