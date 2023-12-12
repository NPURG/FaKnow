from typing import List

import yaml
import torch
from torch.utils.data import DataLoader

from faknow.utils.util import dict2str
from faknow.train.cafe_trainer import CafeTrainer
from faknow.evaluate.evaluator import Evaluator
from faknow.data.dataset.cafe_dataset import CafeDataset
from faknow.model.content_based.multi_modal.cafe import CAFE

__all__ = ['run_cafe', 'run_cafe_from_yaml']


def run_cafe(dataset_dir: str,
             batch_size=64,
             lr=1e-3,
             weight_decay=0,
             epoch_num=100,
             metrics: List = None,
             device="cpu"):
    """
    run CAFE
    Args:
        dataset_dir (str): path of data,including training data and testing data.
        batch_size (int): batch size, default=64
        lr (float): learning rate, default=0.001
        weight_decay (float): weight_decay, default=0
        epoch_num(int): number of epochs, default=50
        metrics (List): evaluation metrics,
            if None, ['accuracy', 'precision', 'recall', 'f1'] is used,
            default=None
        device (str): device to run model, default='cuda:0'
    """
    # ---  Load Data  ---
    train_set = CafeDataset(
        "{}/train_text_with_label.npz".format(dataset_dir),
        "{}/train_image_with_label.npz".format(dataset_dir))
    test_set = CafeDataset("{}/test_text_with_label.npz".format(dataset_dir),
                           "{}/test_image_with_label.npz".format(dataset_dir))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=True)

    model = CAFE()

    optim_task_similarity = torch.optim.Adam(
        model.similarity_module.parameters(), lr=lr, weight_decay=weight_decay)

    sim_params_id = list(map(id, model.similarity_module.parameters()))
    base_params = filter(lambda p: id(p) not in sim_params_id,
                         model.parameters())
    optim_task_detection = torch.optim.Adam(base_params,
                                            lr=lr,
                                            weight_decay=weight_decay)

    evaluator = Evaluator(metrics)

    trainer = CafeTrainer(model,
                          evaluator,
                          optim_task_detection,
                          optim_task_similarity,
                          device=device)

    trainer.fit(train_loader, epoch_num)

    if test_loader is not None:
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_cafe_from_yaml(path: str) -> None:
    """
    run EANN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_cafe(**_config)
