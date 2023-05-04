from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader

from faknow.model.content_based.multi_modal.safe import SAFE
from faknow.data.dataset.safe_dataset import SAFENumpyDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


def run_safe(root):
        dataset = SAFENumpyDataset(root)

        val_size = int(len(dataset) * 0.1)
        test_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = SAFE()
        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                list(model.parameters())),
            lr=0.00025
        )
        evaluator = Evaluator(["accuracy", "precision", "recall", "f1"])

        trainer = BaseTrainer(model, evaluator, optimizer)
        trainer.fit(train_loader=train_loader, num_epoch=100, validate_loader=val_loader, save=True)
        test_result = trainer.evaluate(test_loader)
        print("test result: ", {dict2str(test_result)})


def main():
    root = Path(r"C:\Users\10749\Desktop\FaKnow\dataset\example\SAFE")
    run_safe(root)


if __name__ == '__main__':
    main()
