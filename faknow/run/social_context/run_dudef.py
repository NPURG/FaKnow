from typing import List, Callable
from numpy import ndarray
import os
import torch
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.dudef import DUDEF
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str
from faknow.data.dataset.dudef_dataset import DudefDataset
from faknow.utils.util import EarlyStopping


def run_dudef(data_dir: str,
              embeddings_index: dict[ndarray],
              baidu_arr: Callable,
              dalianligong_arr: Callable,
              boson_value: Callable,
              auxilary_features: Callable,
              MAX_NUM_WORDS=6000,
              epochs=50,
              batch_size=64,
              lr_param=0.01,
              metrics: List = None,
              device='cpu'):
    """
    Args:
        data_dir (str): Root directory where the dataset should be saved
        embeddings_index dict[ndarray]: word vectors,ndarray:(300,), for example:'!':[0.618666,...]
        baidu_arr (Callable): Function to calculate emotions using Baidu API.
        dalianligong_arr (Callable): Function to calculate emotions using Dalian Ligong University's method.
        boson_value (Callable): Function to extract boson features
        auxilary_features (Callable): Function to extract auxiliary features from the text.
        MAX_NUM_WORDS (int): size of senmantics, default=6000
        epochs (int): number of epochs, default=50
        batch_size (int): batch size, default=64
        lr_param (float): learning rate, default=0.01
        metrics (List): evaluation metrics, if None, use default metrics, default=None
        device (str): device, default='cpu'
    """

    Dataset = DudefDataset(data_dir, baidu_arr, dalianligong_arr, boson_value, auxilary_features)
    Dataset.get_label(data_dir)
    Dataset.get_dualemotion(data_dir)
    data_path = os.path.join(data_dir, 'data')
    Dataset.get_senmantics(data_path, MAX_NUM_WORDS, embeddings_index)
    (train_loader, val_loader, test_loader, semantics_embedding_matrix) = (
        Dataset.load_dataset(data_path, batch_size=batch_size,
                             input_types=['emotions', 'semantics']))

    model = DUDEF(input_size=semantics_embedding_matrix.shape[1],
                  emotion_len=train_loader[0].shape[1],
                  hidden_size=32,
                  embedding_matrix=torch.tensor(semantics_embedding_matrix))
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

    if test_loader is not None:
        test_result = trainer.evaluate(test_loader)
        print('test result: ', dict2str(test_result))
