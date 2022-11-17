import os
from time import time
import datetime
from tqdm import tqdm
from tqdm.contrib import tenumerate
from typing import Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from template.evaluate.evaluator import Evaluator
from template.model.model import AbstractModel


class AbstractTrainer:
    def __init__(self, model: AbstractModel, evaluator: Evaluator, criterion: Callable,
                 optimizer: Optimizer):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        """evaluate after training"""
        raise NotImplementedError

    def fit(self, train_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int, validate_data=None,
            validate_size=None, saved=False, save_path=None):
        raise NotImplementedError


class Trainer:
    def __init__(self, model: AbstractModel, evaluator: Evaluator, criterion: Callable,
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
        labels = []
        for X, y in dataloader:
            outputs.append(self.model(X))
            labels.append(y)
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(labels))

        # 或者事先分好批次，把每个batch的output与y作为一个tuple，多个批次的tuple共同组成一个list
        # self.evaluator.evaluate([(self.model(X), y) for X, y in dataloader])

    def _train_epoch(self, data, batch_size: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        dataloader = DataLoader(data, batch_size, shuffle=True)
        # todo tqdm与print冲突
        # data_iter = tenumerate(
        #     dataloader,
        #     total=len(dataloader),
        #     ncols=100,
        #     desc=f'Training',
        #     leave=False
        # )
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

    def fit(self, train_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int, validate_data=None,
            validate_size: Optional[float] = None, saved=False, save_path: Optional[str] = None):
        """training"""
        # whether split validation set
        validation = True
        if validate_data is None and validate_size is None:
            validation = False
        else:
            validate_size = int(validate_size * len(train_data))
            train_size = len(train_data) - validate_size
            train_data, validate_data = torch.utils.data.random_split(
                train_data, [train_size, validate_size])

        # tqdm.write('----start training-----')
        print(f'training data size={len(train_data)}')
        if validation:
            print(f'validation data size={validate_size}')

        # training for epochs
        print('----start training-----')
        for epoch in range(epochs):
            training_start_time = time()
            training_loss = self._train_epoch(train_data, batch_size)
            training_end_time = time()
            print(f'epoch={epoch}, '
                  f'time={training_end_time - training_start_time}s, '
                  f'train loss={training_loss}')
            if validation:
                validate_result = self._validate_epoch(validate_data,
                                                       batch_size)
                print(f'validation result: {validate_result}')
            # tqdm.write(f'epoch={epoch}, '
            #            f'time={training_end_time - training_start_time}s, '
            #            f'train loss={training_loss}')
            # tqdm.write(f'validation result: {validate_result}')

        # save the model
        if saved:
            if save_path is None:
                save_path = os.path.join(os.getcwd(), "save",
                                         f"{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pth")
            torch.save(self.model.state_dict(), save_path)
