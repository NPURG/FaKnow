import json
import logging
import os
import random
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from faknow.data.dataset.m3fend_dataset import M3FENDDataSet
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.m3fend import M3FEND
from faknow.train.m3fend_trainer import M3FENDTrainer
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import EarlyStopping
from faknow.utils.utils_m3fend import Recorder, data2gpu, Averager, metrics, tuple2dict

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False # 将 PyTorch 中 cuDNN 库的自动优化设置为禁用
torch.backends.cudnn.deterministic = True # 将 cuDNN 库的算法设置为确定性的
# 在深度学习实验中，尤其是在涉及卷积神经网络（CNN）等使用 cuDNN 的模型时，禁用 cuDNN 的自动优化并启用确定性算法是为了确保实验结果的一致性，使得实验可复现。


def _init_fn(worker_id):
    np.random.seed(2021)


def getFileLogger(log_file):
    logger = logging.getLogger()
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
    if device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # 将 CUDA_VISIBLE_DEVICES 环境变量设置为在命令行参数中指定的 GPU 的值
    elif device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = '' # 将 CUDA_VISIBLE_DEVICES 环境变量设置为'', 即cpu
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

    # todo early_stopping
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
    # trainer.fit(train_loader=train_loader, num_epochs=epochs, validate_loader=val_loader)


if __name__ == '__main__':
    run_m3fend(dataset='en', domain_num=3, lr=0.0001, device='cuda', gpu='0')