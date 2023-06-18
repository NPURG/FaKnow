from typing import List, Dict, Any

import torch
import yaml
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.bigcn import BiGCN
from faknow.train.base_gnn_trainer import BaseGNNTrainer
from faknow.utils.util import DropEdge

__all__ = ['run_bigcn', 'run_bigcn_from_yaml']


def run_bigcn(root: str,
              name: str,
              feature: str,
              splits: List[str] = None,
              batch_size=128,
              epochs=45,
              hidden_size=128,
              td_drop_rate=0.2,
              bu_drop_rate=0.2,
              lr=0.01,
              weight_decay=0.001,
              metrics: List = None,
              device='cpu'):
    if splits is None:
        splits = ['train', 'val', 'test']

    train_dataset = UPFD(root, name, feature, 'train',
                         DropEdge(td_drop_rate, bu_drop_rate))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    if 'val' in splits:
        val_dataset = UPFD(root, name, feature, 'val',
                           DropEdge(td_drop_rate, bu_drop_rate))
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    else:
        val_loader = None

    feature_size = train_dataset.num_features
    model = BiGCN(feature_size, hidden_size, hidden_size)
    bu_params_id = list(map(id, model.BURumorGCN.parameters()))
    base_params = filter(lambda p: id(p) not in bu_params_id,
                         model.parameters())
    optimizer = torch.optim.Adam([{
        'params': base_params
    }, {
        'params': model.BURumorGCN.parameters(),
        'lr': lr / 5
    }],
        lr=lr,
        weight_decay=weight_decay)
    evaluator = Evaluator(metrics)

    trainer = BaseGNNTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, epochs, val_loader)

    if 'test' in splits:
        test_dataset = UPFD(root, name, feature, 'test',
                            DropEdge(td_drop_rate, bu_drop_rate))
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f'test result={test_result}')


def run_bigcn_from_yaml(config: Dict[str, Any]):
    run_bigcn(**config)


if __name__ == '__main__':
    with open(r'..\..\properties\upfd.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_bigcn_from_yaml(_config)
