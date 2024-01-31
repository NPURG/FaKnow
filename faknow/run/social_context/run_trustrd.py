from typing import List, Optional, Dict
import copy

import yaml
import torch

from faknow.model.social_context.trustrd import TrustRDPreTrainModel, TrustRD
from faknow.train.base_gnn_trainer import BaseGNNTrainer
from faknow.data.dataset.trustrd_dataset import TrustRDDataset
from torch_geometric.loader import DataLoader
from faknow.utils.util import dict2str
from faknow.evaluate.evaluator import Evaluator

__all__ = ['run_trustrd', 'run_trustrd_from_yaml']


def run_trustrd(train_data: List,
                data_path: str,
                tree_dic: Dict,
                val_data: Optional[List] = None,
                test_data: Optional[List] = None,
                sigma_m=0.1,
                eta=0.4,
                zeta=0.02,
                drop_rate=0.4,
                input_feature=192,
                hidden_feature=64,
                num_classes=4,
                batch_size=128,
                pre_train_epoch=25,
                epochs=200,
                net_hidden_dim=64,
                net_gcn_layers=3,
                lr=0.0005,
                weight_decay=1e-4,
                metrics: Optional[List] = None,
                device='cpu'):
    """
    Args:
        train_data(List): index list of training nodes.
        tree_dic(Dict): the dictionary of graph edge.
        data_path(str): path of data doc.
        val_data(Optional[List]): index list of validation nodes, default=None
        test_data(Optional[List]): index list of test nodes, default=None
        sigma_m(float): data perturbation Standard Deviation. default=0.1
        eta(float): data perturbation weight. default=0.4.
        zeta(float): parameter perturbation weight. default=0.02
        drop_rate(float): drop rate of edge. default=0.4.
        input_feature(int): the feature size of input. default=192.
        hidden_feature(int): the feature size of hidden embedding. default=64.
        num_classes(int): the num of class. default=4.
        batch_size(int): batch size. default=128.
        pre_train_epoch(int): pretrained epoch num. default=25.
        epochs(int): epoch num. default=200.
        net_hidden_dim(int): the feature size of hidden embedding. defult=64.
        net_gcn_layers(int): the gcn encoder layer num. default=3.
        lr(float): learning rate. default=0.0005.
        weight_decay(float): weight decay. default=1e-4.
        metrics (List): metrics for evaluation, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        device(str): device. default='cpu'.
    """

    unsup_model = TrustRDPreTrainModel(net_hidden_dim, net_gcn_layers)
    unsup_optimizer = torch.optim.Adam(unsup_model.parameters(),
                                       lr=lr,
                                       weight_decay=weight_decay)

    pre_train_set = TrustRDDataset(train_data,
                                   tree_dic,
                                   drop_rate=0.2,
                                   data_path=data_path)
    pre_train_loader = DataLoader(pre_train_set,
                                  batch_size=batch_size,
                                  shuffle=True)

    evaluator = Evaluator(metrics)
    pre_trainer = BaseGNNTrainer(unsup_model,
                                 evaluator,
                                 unsup_optimizer,
                                 device=device)
    pre_trainer.fit(pre_train_loader,
                    num_epochs=pre_train_epoch,
                    validate_loader=None)
    unsup_model.eval()

    pre_trained_model = copy.deepcopy(unsup_model)
    model = TrustRD(pre_trained_model, input_feature, hidden_feature,
                    num_classes, sigma_m, eta, zeta)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    train_set = TrustRDDataset(train_data,
                               tree_dic,
                               drop_rate=drop_rate,
                               data_path=data_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if val_data is not None:
        val_set = TrustRDDataset(val_data, tree_dic, data_path=data_path)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    trainer = BaseGNNTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, num_epochs=epochs, validate_loader=val_loader)

    if test_data is not None:
        test_set = TrustRDDataset(test_data, tree_dic, data_path=data_path)
        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_trustrd_from_yaml(path: str):
    """
    run TrustRD from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_trustrd(**_config)
