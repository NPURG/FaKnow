import json
import logging
import os
import random
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
from faknow.utils.utils_m3fend import Recorder, data2gpu, Averager, metrics

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
        use_cuda: bool = True,
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
        save_param_dir: str = './param_model',
        param_log_dir: str = './logs/param',
        early_stop: int = 3,
        epochs: int = 50,
        device: str = 'cpu',
        gpu: int = 0,
        metrics: List = None,
):
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
    # 设置一个文件日志器（logger）
    if not os.path.exists(param_log_dir):
        os.makedirs(param_log_dir)
    param_log_file = os.path.join(param_log_dir, 'm3fend' + '_' + 'oneloss_param.txt')
    logger = getFileLogger(param_log_file)

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

    # loss
    # 在模型的 calculate_loss 函数中体现

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

    # evaluator
    evaluator = Evaluator(metrics)

    # todo early_stopping
    early_stopping = EarlyStopping(patience=3)

    # 设置模型为训练模式
    model.train()

    # 创建进度条对象
    train_data_iter = tqdm(train_loader)

    # 循环遍历训练数据 : 将数据移动到GPU + 将所有样本的归一化特征按照它们的领域（domain）信息保存在self.all_feature 字典中
    for step_n, batch in enumerate(train_data_iter):
        batch_data = data2gpu(batch, use_cuda)
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
    trainer.fit(train_loader=train_loader, num_epochs=epochs, validate_loader=val_loader)


"""    trainer = M3FENDTrainer(emb_dim=emb_dim, mlp_dims=mlp_dims, use_cuda=use_cuda, lr=lr,
                            train_loader=train_loader, dropout=dropout, weight_decay=weight_decay,
                            val_loader=val_loader, test_loader=test_loader, category_dict=category_dict,
                            early_stop=early_stop, epochs=epochs,
                            save_param_dir=os.path.join(save_param_dir, 'm3fend'),
                            semantic_num=semantic_num, emotion_num=emotion_num, style_num=style_num,
                            lnn_dim=lnn_dim, dataset=dataset)

    # 简单的参数搜索过程，对于给定的参数进行多次训练，记录并比较它们的性能，最终输出最佳的参数配置和相应的模型
    train_param = {
        'lr': [lr] * 10
    }
    print(train_param)

    param = train_param
    best_param = []  # 用于存储每个参数的最佳取值

    json_path = './logs/json/' + 'm3fend.json'
    json_result = []
    for p, vs in param.items():  # param是一个字典，键：参数名称，值：参数可能取值的列表
        best_metric = {}
        best_metric['metric'] = 0
        best_v = vs[0]
        best_model_path = None
        for i, v in enumerate(vs):
            setattr(trainer, p, v)
            metrics, model_path = trainer.train(logger)
            json_result.append(metrics)
            if (metrics['metric'] > best_metric['metric']):
                best_metric = metrics
                best_v = v
                best_model_path = model_path
        best_param.append({p: best_v})

        print("best model path:", best_model_path)
        print("best metric:", best_metric)
        logger.info("best model path:" + best_model_path)
        logger.info("best param " + p + ": " + str(best_v))
        logger.info("best metric:" + str(best_metric))
        logger.info('--------------------------------------\n')
    with open(json_path, 'w') as file:
        json.dump(json_result, file, indent=4, ensure_ascii=False)"""




if __name__ == '__main__':
    run_m3fend(dataset='en', domain_num=3, lr=0.0001, gpu=0)