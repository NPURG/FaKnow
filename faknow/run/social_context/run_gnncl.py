from typing import List

import torch
import yaml
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DenseDataLoader
from torch_geometric.transforms import ToUndirected, ToDense

from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.gnncl import GNNCL
from faknow.train.dense_gnn_trainer import DenseGNNTrainer

__all__ = ['run_gnncl', 'run_gnncl_from_yaml']


def run_gnncl(root: str,
              name: str,
              feature: str,
              splits=None,
              batch_size=128,
              max_nodes=500,
              lr=0.1,
              epochs=70,
              metrics: List = None,
              device='cpu'):
    if splits is None:
        splits = ['train', 'val', 'test']

    train_dataset = UPFD(root,
                         name,
                         feature,
                         'train',
                         transform=ToDense(max_nodes),
                         pre_transform=ToUndirected())
    train_loader = DenseDataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    if 'val' in splits:
        val_dataset = UPFD(root,
                           name,
                           feature,
                           'val',
                           transform=ToDense(max_nodes),
                           pre_transform=ToUndirected())
        val_loader = DenseDataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
    else:
        val_loader = None

    model = GNNCL(train_dataset.num_features, max_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    evaluator = Evaluator(metrics)

    trainer = DenseGNNTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, epochs, val_loader)

    if 'test' in splits:
        test_dataset = UPFD(root,
                            name,
                            feature,
                            'test',
                            transform=ToDense(max_nodes),
                            pre_transform=ToUndirected())
        test_loader = DenseDataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f'test result={test_result}')


def run_gnncl_from_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_gnncl(**_config)
