from time import time

import torch
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from evaluator.evaluator import Evaluator
from model.model import Model
from torch.utils.data.dataloader import DataLoader


class Trainer:
    def __init__(self, model: Model, evaluator: Evaluator, criterion: _Loss,
                 optimizer: Optimizer):
        self.model = model
        self.criterion = criterion  # todo 放入trainer or model
        self.optimizer = optimizer
        self.evaluator = evaluator

    @torch.no_grad()
    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        """evaluate after training"""
        self.model.eval()
        dataloader = DataLoader(data, batch_size, shuffle=True)
        # todo
        # 不分批次，采用concat直接把每批的output与y组合在一起，再分别传入
        outputs = []
        Y = []
        for X, y in dataloader:
            outputs.append(self.model(X))
            Y.append(y)
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(Y))

        # 或者事先分好批次，把每个batch的output与y作为一个tuple，多个批次的tuple共同组成一个list
        # self.evaluator.evaluate([(self.model(X), y) for X, y in dataloader])

    def _train_epoch(self, data, batch_size: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        dataloader = DataLoader(data, batch_size, shuffle=True)
        for batch_id, (X, y) in enumerate(dataloader):
            # forward
            output = self.model(X)
            loss = self.criterion(output, y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def _validate_epoch(self, data, batch_size: int):
        """validation after training for one epoch"""
        return self.evaluate(data, batch_size)

    def fit(self, train_data: torch.utils.data.Dataset,
            validate_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int):
        """training"""
        # todo 使用validation size
        print('----start training-----')
        for epoch in range(epochs):
            training_start_time = time()
            training_loss = self._train_epoch(train_data, batch_size)
            training_end_time = time()
            validate_result = self._validate_epoch(validate_data, batch_size)
            print(f'epoch={epoch}, '
                  f'training_time={training_end_time - training_start_time}s, '
                  f'training loss={training_loss}')
            print(f'validation result: {validate_result}')
