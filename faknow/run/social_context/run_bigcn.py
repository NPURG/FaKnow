from typing import List

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
              feature='bert',
              splits: List[str] = None,
              batch_size=128,
              epochs=50,
              hidden_size=128,
              td_drop_rate=0.2,
              bu_drop_rate=0.2,
              lr=0.01,
              weight_decay=0.001,
              metrics: List = None,
              device='cpu'):
    r"""
    run BiGCN using UPFD dataset, including training, validation and testing.
    If validation and testing data are not provided, only training is performed.

    Args:
        root (str): Root directory where the dataset should be saved
        name (str): The name of the graph set (:obj:`"politifact"`, :obj:`"gossipcop"`)
        feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`, :obj:`"bert"`, :obj:`"content"`)
            If set to :obj:`"profile"`, the 10-dimensional node feature
            is composed of ten Twitter user profile attributes.
            If set to :obj:`"spacy"`, the 300-dimensional node feature is
            composed of Twitter user historical tweets encoded by
            the `spaCy word2vec encoder
            <https://spacy.io/models/en#en_core_web_lg>`_.
            If set to :obj:`"bert"`, the 768-dimensional node feature is
            composed of Twitter user historical tweets encoded by the
            `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
            If set to :obj:`"content"`, the 310-dimensional node feature is
            composed of a 300-dimensional "spacy" vector plus a
            10-dimensional "profile" vector. default='bert'
        splits (List[str]): dataset split, including 'train', 'val' and 'test'.
            If None, ['train', 'val', 'test'] will be used. Default=None
        batch_size (int): batch size, default=128
        epochs (int): number of epochs, default=45
        hidden_size (int): dimension of hidden layer, default=128
        td_drop_rate (float): drop rate of drop edge in top-down direction, default=0.2
        bu_drop_rate (float): drop rate of drop edge in bottom-up direction, default=0.2
        lr (float): learning rate, default=0.01
        weight_decay (float): weight decay, default=0.001
        metrics (List): evaluation metrics, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        device (str): device, default='cpu'
    """

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


def run_bigcn_from_yaml(path: str):
    """
    run BiGCN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_bigcn(**_config)
