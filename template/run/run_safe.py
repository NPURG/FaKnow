import torch
from pathlib import Path
from template.evaluate.evaluator import Evaluator
from template.model.multi_modal.safe import SAFE
from template.data.dataset.safe_dataset import SAFEDataset, SAFENumpyDataset
from template.train.trainer import BaseTrainer
from template.utils.sif_embedding import sif_embedding


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
    root =Path("C:\\Users\\10749\\Desktop\\embedding")
    run_safe(root)

