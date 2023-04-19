import torch
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

from faknow.data.dataset.nep_dataset import NEPDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.nep import NEP
from faknow.train.trainer import BaseTrainer


def run_nep():
    data = torch.load(r'F:\code\python\News-Environment-Perception\test\val_dataset.pt')
    dataset = NEPDataset(*data)

    train_size = int(0.01 * len(dataset))
    val_size = int(0.02 * train_size)
    _, train_set, val_set = random_split(dataset,
                                         [len(dataset) - train_size - val_size, train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    model = NEP(bert='bert-base-chinese')
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=5e-4)
    evaluator = Evaluator()
    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, 10, val_loader)


if __name__ == '__main__':
    run_nep()
