from typing import Dict, Any, List

import torch
import yaml
from torch.utils.data import random_split, DataLoader

from faknow.data.dataset.safe_dataset import SAFENumpyDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.safe import SAFE
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


def run_safe(
        root: str,
        batch_size=64,
        lr=0.00025,
        metrics: List = None,
        num_epochs=100
):
    dataset = SAFENumpyDataset(root)

    val_size = int(len(dataset) * 0.1)
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = SAFE()
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(model.parameters())),
        lr
    )
    evaluator = Evaluator(metrics)

    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader=train_loader, num_epochs=num_epochs, validate_loader=val_loader)
    test_result = trainer.evaluate(test_loader)
    print("test result: ", {dict2str(test_result)})


def run_safe_from_yaml(config: Dict[str, Any]):
    run_safe(**config)


if __name__ == '__main__':
    with open(r'..\properties\safe.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_safe_from_yaml(_config)
