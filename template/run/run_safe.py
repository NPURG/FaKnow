from pathlib import Path

import torch

from template.data.dataset.safe_dataset import SAFENumpyDataset
from template.evaluate.evaluator import Evaluator
from template.model.multi_modal.safe import SAFE
from template.train.trainer import BaseTrainer


def run_safe(root: str):
    # dataset = SAFEDataset(
    #     root=root,
    #     embedding=sif_embedding,
    #     max_len=16,
    # )
    dataset = SAFENumpyDataset(root)

    model = SAFE()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=0.00025)
    evaluator = Evaluator(["accuracy", "precision", "recall", "f1"])

    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(dataset,
                batch_size=100,
                epochs=100,
                validate_size=0.2,
                saved=True)


if __name__ == '__main__':
    # root = "E:\\Python_program\\Template\\dataset\\example\\dataset_example_SAFE"
    # root = Path("E:\\Python_program\\SAFE\\embedding")
    root =Path("F:\\code\\python\\SAFE-pytorch\\embedding")
    run_safe(root)

