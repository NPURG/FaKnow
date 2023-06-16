from typing import List, Dict, Any

import torch
import yaml
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.gcnfn import GCNFN
from faknow.train.base_gnn_trainer import BaseGNNTrainer

__all__ = ['run_gcnfn', 'run_gcnfn_from_yaml']


def run_gcnfn(root: str,
              name: str,
              feature: str,
              splits=None,
              batch_size=128,
              epochs=110,
              hidden_size=128,
              lr=0.001,
              weight_decay=0.01,
              metrics: List = None):
    if splits is None:
        splits = ['train', 'val', 'test']

    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    if 'val' in splits:
        val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    else:
        val_loader = None

    model = GCNFN(train_dataset.num_features, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    evaluator = Evaluator(metrics)

    trainer = BaseGNNTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, epochs, val_loader)

    if 'test' in splits:
        test_dataset = UPFD(root, name, feature, 'test', ToUndirected())
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f'test result={test_result}')


def run_gcnfn_from_yaml(config: Dict[str, Any]):
    run_gcnfn(**config)


if __name__ == '__main__':
    with open(r'..\..\properties\upfd.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_gcnfn_from_yaml(_config)
