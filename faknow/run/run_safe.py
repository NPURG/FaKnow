from typing import Dict, Any, List, Optional

import torch
import yaml
from torch.utils.data import random_split, DataLoader

from faknow.data.dataset.safe_dataset import SAFENumpyDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.safe import SAFE
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


def run_safe(
        train_path: str,
        validate_path: str = None,
        test_path: str = None,
        embedding_size: int = 300,
        conv_in_size: int = 32,
        filter_num: int = 128,
        cnn_out_size: int = 200,
        dropout: float = 0.,
        loss_weights: Optional[List[float]] = None,
        batch_size=64,
        lr=0.00025,
        metrics: List = None,
        num_epochs=100,
        device='cpu',
):
    train_dataset = SAFENumpyDataset(train_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    if validate_path is not None:
        val_dataset = SAFENumpyDataset(validate_path)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = None

    model = SAFE(embedding_size, conv_in_size, filter_num, cnn_out_size, dropout, loss_weights)
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(model.parameters())),
        lr
    )
    evaluator = Evaluator(metrics)

    trainer = BaseTrainer(model=model, evaluator=evaluator, optimizer=optimizer, device=device)
    trainer.fit(train_loader=train_loader, num_epochs=num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = SAFENumpyDataset(test_path)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def run_safe_from_yaml(config: Dict[str, Any]):
    run_safe(**config)


if __name__ == '__main__':
    with open(r'..\properties\safe.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_safe_from_yaml(_config)
