from typing import List
import os
import torch
import yaml
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.dudef import DUDEF
from faknow.train.trainer import BaseTrainer
from torch.utils.data import DataLoader
from faknow.utils.util import dict2str
from faknow.data.dataset.dudef_dataset import DudefDataset
from faknow.utils.util import EarlyStopping

__all__ = ['run_dudef', 'run_dudef_from_yaml']


def run_dudef(data_dir: str,
              embeddings_index,
              MAX_NUM_WORDS = 6000,
              epochs = 50,
              batch_size = 64,
              lr_param = 0.01,
              metrics: List = None,
              device='cpu'):

    """
       run DUDEF

       Args:
           data_dir(str): Root directory where the dataset should be saved
           embeddings_index(torch.Tensor): word vectors
           MAX_NUM_WORDS(int): size of senmantics, default=6000
           epochs (int): number of epochs, default=50
           batch_size (int): batch size, default=64
           lr_param (float): learning rate, default=0.01
           metrics (List): evaluation metrics, if None, use default metrics, default=None
           device (str): device, default='cpu'
    """

    DudefDataset.get_label(data_dir)
    DudefDataset.get_dualemotion(data_dir)
    data_path = os.path.join(data_dir, 'data')
    DudefDataset.get_senmantics(data_path,MAX_NUM_WORDS,embeddings_index)
    (train_data, val_data, test_data, train_label, val_label, test_label,
     data, label, semantics_embedding_matrix) = DudefDataset.load_dataset(
        data_path, input_types=['emotions','semantics'])
    train_dataset = {}
    for i in range(len(train_data[0])):
        train_dataset[i] = {}
        train_dataset[i]['data'] = {}
        train_dataset[i]['data']['emotions'] = train_data[0][i]
        train_dataset[i]['data']['senmantics'] = train_data[1][i]
        train_dataset[i]['label'] = train_label[i][1]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = {}
    for i in range(len(val_data[0])):
        val_dataset[i] = {}
        val_dataset[i]['data'] = {}
        val_dataset[i]['data']['emotions'] = val_data[0][i]
        val_dataset[i]['data']['senmantics'] = val_data[1][i]
        val_dataset[i]['label'] = val_label[i][1]
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)

    test_dataset = {}
    for i in range(len(test_data[0])):
        test_dataset[i] = {}
        test_dataset[i]['data'] = {}
        test_dataset[i]['data']['emotions'] = test_data[0][i]
        test_dataset[i]['data']['senmantics'] = test_data[1][i]
        test_dataset[i]['label'] = test_label[i][1]
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    model = DUDEF(input_size = semantics_embedding_matrix.shape[1],
                  emotion_len = train_data[0].shape[1],
                  hidden_size=32,
                  embedding_matrix = torch.tensor(semantics_embedding_matrix))
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr_param)
    evaluator = Evaluator(metrics)

    stopping = EarlyStopping(patience=10)
    trainer = BaseTrainer(model,
                          evaluator,
                          optimizer,
                          device=device,
                          early_stopping=stopping)

    trainer.fit(train_loader, epochs, validate_loader=val_loader)

    if test_data is not None:
        test_result = trainer.evaluate(test_loader)
        print('test result: ', dict2str(test_result))


def run_dudef_from_yaml(path: str):
    """
    run DUDEF from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_dudef(**_config)