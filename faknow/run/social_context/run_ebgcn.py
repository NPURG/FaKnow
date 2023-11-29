from typing import Optional, Dict, List

import torch
import yaml
from torch_geometric.loader import DataLoader

from faknow.model.social_context.ebgcn import EBGCN
from faknow.data.dataset.bigcn_dataset import BiGCNDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.train.base_gnn_trainer import BaseGNNTrainer
from faknow.utils.util import dict2str

__all__ = ['run_ebgcn', 'run_ebgcn_from_yaml']


def run_ebgcn(train_data: List,
              val_data: Optional[list] = None,
              test_data: Optional[list] = None,
              data_path=None,
              tree_dic: Dict = None,
              batch_size=128,
              input_size=5000,
              hidden_size=64,
              output_size=64,
              edge_num=2,
              dropout=0.5,
              num_class=4,
              edge_loss_weight=0.2,
              lr=0.0005,
              weight_decay=0.1,
              lr_scale_bu=5,
              lr_scale_td=1,
              metrics=None,
              num_epochs=200,
              device='cpu'):
    """
    run EBGCN, including training, validation and testing.
    If validate_path and test_path are None, only training is performed.

    Args:
        train_data(list): index list of training nodes.
        val_data(Optional[list]): index list of validation nodes.
        test_data(Optional[list]): index list of test nodes.
        data_path(str): path of data doc. default=None
        tree_dic(dict): the dictionary of graph edge.
        batch_size(int): batch size. default=128.
        input_size(int): the feature size of input. default=5000.
        hidden_size(int): the feature size of hidden embedding. default=64.
        output_size(int): the feature size of output embedding. default=64.
        edge_num(int): the num of edge type. default=2.
        dropout(float): dropout rate. default=0.5.
        num_class(int): the num of output type. default=4
        edge_loss_weight(float): the weight of edge loss. default=0.2.
        lr(float): learning rate. default=0.0005.
        weight_decay(float): weight decay. default=0.1.
        lr_scale_bu(int): learning rate scale for down-top direction. default=5.
        lr_scale_td(int): learning rate scale for top-down direction. default=1.
        metrics (List): metrics for evaluation, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        num_epochs(int): epoch num. default=200.
        device(str): device. default='cpu'.
    """
    train_set = BiGCNDataset(train_data, tree_dic, data_path)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)
    if val_data is not None:
        val_set = BiGCNDataset(val_data, tree_dic, data_path)
        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=False)
    else:
        val_loader = None

    model = EBGCN(input_size, hidden_size, output_size, edge_num, dropout,
                  num_class, edge_loss_weight)

    TD_params = list(map(id, model.TDRumorGCN.conv1.parameters()))
    TD_params += list(map(id, model.TDRumorGCN.conv2.parameters()))
    BU_params = list(map(id, model.BURumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BURumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params + TD_params,
                         model.parameters())
    optimizer = torch.optim.Adam([{
        'params': base_params
    }, {
        'params': model.BURumorGCN.conv1.parameters(),
        'lr': lr / lr_scale_bu
    }, {
        'params': model.BURumorGCN.conv2.parameters(),
        'lr': lr / lr_scale_bu
    }, {
        'params': model.TDRumorGCN.conv1.parameters(),
        'lr': lr / lr_scale_td
    }, {
        'params': model.TDRumorGCN.conv2.parameters(),
        'lr': lr / lr_scale_td
    }],
                                 lr=lr,
                                 weight_decay=weight_decay)

    evaluator = Evaluator(metrics)
    trainer = BaseGNNTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader,
                num_epochs=num_epochs,
                validate_loader=val_loader)

    if test_data is not None:
        test_set = BiGCNDataset(test_data, tree_dic, data_path=data_path)
        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=30)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_ebgcn_from_yaml(path: str):
    """
    run EBGCN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_ebgcn(**_config)
