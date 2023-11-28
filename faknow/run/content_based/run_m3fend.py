import os
import random
import sys
from typing import List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from faknow.data.dataset.m3fend_dataset import M3FENDDataSet
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.m3fend import M3FEND
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import EarlyStopping
from faknow.utils.util import data2gpu

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False # 将 PyTorch 中 cuDNN 库的自动优化设置为禁用
torch.backends.cudnn.deterministic = True # 将 cuDNN 库的算法设置为确定性的
# 在深度学习实验中，尤其是在涉及卷积神经网络（CNN）等使用 cuDNN 的模型时，禁用 cuDNN 的自动优化并启用确定性算法是为了确保实验结果的一致性，使得实验可复现。

__all__ = ['_init_fn', 'run_m3fend', 'run_m3fend_from_yaml']


def _init_fn(worker_id):
    np.random.seed(2021)


def run_m3fend(
        dataset: str = 'ch',
        domain_num: int = 3,
        emb_dim: int = 768,
        mlp_dims: list = [384],
        batch_size: int = 64,
        num_workers: int = 4,
        max_len: int = 170,
        lr: float = 0.0001,
        dropout: float = 0.2,
        weight_decay: float = 0.00005,
        semantic_num: int = 7,
        emotion_num: int = 7,
        style_num: int = 2,
        lnn_dim: int = 50,
        early_stop: int = 3,
        epochs: int = 50,
        device: str = 'gpu',
        gpu: str = '',
        metrics: List = None,
):
    """
    Train and evaluate the M3FEND model.

    Args:
        dataset (str, optional): Dataset name. Defaults to 'ch'.
        domain_num (int, optional): Number of domains. Defaults to 3.
        emb_dim (int, optional): Dimension of the embeddings. Defaults to 768.
        mlp_dims (list, optional): List of dimensions for the MLP layers. Defaults to [384].
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        max_len (int, optional): Maximum sequence length. Defaults to 170.
        lr (float, optional): Learning rate. Defaults to 0.0001.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        weight_decay (float, optional): Weight decay for optimization. Defaults to 0.00005.
        semantic_num (int, optional): Number of semantic categories. Defaults to 7.
        emotion_num (int, optional): Number of emotion categories. Defaults to 7.
        style_num (int, optional): Number of style categories. Defaults to 2.
        lnn_dim (int, optional): Dimension of the latent narrative space. Defaults to 50.
        early_stop (int, optional): Number of epochs for early stopping. Defaults to 3.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        device (str, optional): Device to run the training on ('cuda' or 'gpu'). Defaults to 'gpu'.
        gpu (str, optional): GPU device ID. Defaults to an empty string.
        metrics (List, optional): List of evaluation metrics. Defaults to None.
    """
    if device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    elif device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ''  # cpu
        if gpu != '':
            print("The current environment is a CPU environment, and the 'gpu' parameter should be removed; Or if you want"
                  " to use the cuda environment, set the 'device' parameter to 'cuda' and specify the 'gpu' parameter as a certain number")
            sys.exit()
    if dataset == 'en':
        root_path = '../../../dataset/example/M3FEND/en/'
        category_dict = {
            "gossipcop": 0,
            "politifact": 1,
            "COVID": 2,
        }
    elif dataset == 'ch':
        root_path = '../../../dataset/example/M3FEND/ch/'
        if domain_num == 9:
            category_dict = {
                "科技": 0,
                "军事": 1,
                "教育考试": 2,
                "灾难事故": 3,
                "政治": 4,
                "医药健康": 5,
                "财经商业": 6,
                "文体娱乐": 7,
                "社会生活": 8,
            }
        elif domain_num == 6:
            category_dict = {
                "教育考试": 0,
                "灾难事故": 1,
                "医药健康": 2,
                "财经商业": 3,
                "文体娱乐": 4,
                "社会生活": 5,
            }
        elif domain_num == 3:
            category_dict = {
                "政治": 0,  # 852
                "医药健康": 1,  # 1000
                "文体娱乐": 2,  # 1440
            }

    print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}; domain_num: {}'.format(lr, 'm3fend',
                                                                                             batch_size, epochs,
                                                                                             gpu, domain_num))
    torch.backends.cudnn.enabled = False

    train_path = root_path + 'train.pkl'
    val_path = root_path + 'val.pkl'
    test_path = root_path + 'test.pkl'

    # dataset & dataloader
    train_dataset = M3FENDDataSet(train_path, max_len, category_dict, dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=_init_fn
    )

    val_dataset = M3FENDDataSet(val_path, max_len, category_dict, dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=_init_fn
    )

    test_dataset = M3FENDDataSet(test_path, max_len, category_dict, dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=_init_fn
    )

    # model
    model = M3FEND(emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, lnn_dim, len(category_dict), dataset)
    if device == 'cuda':
        model = model.cuda()

    # loss
    # 在模型的 calculate_loss 函数中体现

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

    # evaluator
    evaluator = Evaluator(metrics)

    # early_stopping
    early_stopping = EarlyStopping(patience=early_stop)

    # 设置模型为训练模式
    model.train()

    # 循环遍历训练数据 : 将数据移动到GPU + 将所有样本的归一化特征按照它们的领域（domain）信息保存在self.all_feature 字典中
    train_data_iter = tqdm(train_loader, desc='Moving data to device')
    for step_n, batch in enumerate(train_data_iter):
        batch_data = data2gpu(batch, device)
        label_pred = model.save_feature(**batch_data)

    # Domain Event Memory 初始化
    model.init_memory()
    print('initialization finished')

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        early_stopping=early_stopping,
        device=device
    )
    trainer.fit(train_loader=train_loader, validate_loader=val_loader, num_epochs=epochs)


def run_m3fend_from_yaml(path: str):
    """
    Load M3FEND configuration from YAML file and run the training and evaluation.

    Args:
        path (str): Path to the YAML configuration file.
    """
    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_m3fend(**_config)
