from typing import List, Optional, Dict

import torch
import yaml
from torch_geometric.loader import DataLoader

from data.dataset.bigcn_dataset import BiGCNDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.bigcn import BiGCN
from faknow.train.base_gnn_trainer import BaseGNNTrainer
from faknow.utils.util import dict2str

__all__ = ['run_bigcn', 'run_bigcn_from_yaml']


def run_bigcn(train_data: List,
              val_data: Optional[list] = None,
              test_data: Optional[list] = None,
              data_path=None,
              tree_dic: Dict = None,
              batch_size=128,
              epochs=200,
              feature_size=5000,
              hidden_size=64,
              output_size=64,
              td_drop_rate=0.2,
              bu_drop_rate=0.2,
              lower=2,
              upper=100000,
              lr=0.0005,
              weight_decay=0.0001,
              metrics: List = None,
              device='cpu'):
    r"""
    run BiGCN, including training, validation and testing.
    If validation and testing data are not provided,
    only training is performed.

    Args:
        train_data(list): index list of training nodes.
        val_data(Optional[list]): index list of validation nodes.
        test_data(Optional[list]): index list of test nodes.
        data_path(str): path of data doc. default=None
        tree_dic(dict): the dictionary of graph edge.
        batch_size(int): batch size. default=128.
        epochs(int): epoch num. default=200.
        feature_size(int): the feature size of input. default=5000.
        hidden_size(int): the feature size of hidden embedding in RumorGCN.
            default=64.
        output_size(int): the feature size of output embedding in RumorGCN.
            default=64.
        td_drop_rate(float): the dropout rate of TDgraph. default=0.2.
        bu_drop_rate(float): the dropout rate of BUgraph. default=0.2.
        lower (int): the minimum of graph size. default=2.
        upper (int): the maximum of graph size. default=100000.
        lr(float): learning rate. default=0.0005.
        weight_decay(float): weight decay. default=0.0001.
        metrics (List): metrics for evaluation,
            if None, ['accuracy', 'precision', 'recall', 'f1'] is used,
            default=None
        device(str): device. default='cpu'.
    """

    train_set = BiGCNDataset(train_data,
                             tree_dic,
                             data_path,
                             lower,
                             upper,
                             td_drop_rate,
                             bu_drop_rate)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if val_data is not None:
        val_set = BiGCNDataset(val_data,
                               tree_dic,
                               data_path,
                               lower,
                               upper,
                               td_drop_rate,
                               bu_drop_rate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    model = BiGCN(feature_size, hidden_size, output_size)
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

    if test_data is not None:
        test_set = BiGCNDataset(test_data,
                                tree_dic,
                                data_path,
                                lower,
                                upper,
                                td_drop_rate,
                                bu_drop_rate)
        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_bigcn_from_yaml(path: str):
    """
    run BiGCN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_bigcn(**_config)
