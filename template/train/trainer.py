import datetime
import os
from time import time
from typing import Optional, Callable, Tuple

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from template.evaluate.evaluator import Evaluator
from template.model.model import AbstractModel
from template.utils.util import dict2str


class AbstractTrainer:
    def __init__(self, model: AbstractModel, evaluator: Evaluator,
                 optimizer: Optimizer, scheduler: Optional[_LRScheduler] = None, loss_func: Optional[Callable] = None):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler

    @torch.no_grad()
    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        """evaluate after training"""
        raise NotImplementedError

    def fit(self, train_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int, validate_data=None,
            validate_size=None, saved=False, save_path=None):
        raise NotImplementedError


class BaseTrainer(AbstractTrainer):
    def __init__(self, model: AbstractModel, evaluator: Evaluator,
                 optimizer: Optimizer, scheduler: Optional[_LRScheduler] = None, loss_func: Optional[Callable] = None):
        super(BaseTrainer, self).__init__(model, evaluator, optimizer, scheduler, loss_func)

    @torch.no_grad()
    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        self.model.eval()
        dataloader = DataLoader(data, batch_size, shuffle=False)
        outputs = []
        labels = []
        for batch_data in dataloader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data['label'])
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(labels))

    def _train_epoch(self, data, batch_size: int, epoch: int) -> torch.float:
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
        msg = None
        for batch_id, batch_data in enumerate(dataloader):
            # using trainer.loss_func first or model.calculate_loss
            if self.loss_func is None:
                result = self.model.calculate_loss(batch_data)
                print(f"result = {result}")
                if type(result) is tuple:
                    loss, msg = result
                else:
                    loss = result
            else:
                loss = self.loss_func(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if msg is not None:
            print(msg)
        return loss

    def _validate_epoch(self, data, batch_size: int):
        """validation after training for one epoch"""
        # todo 计算best score，作为最佳结果保存
        return self.evaluate(data, batch_size)

    def _split_train_validate(self, train_data: torch.utils.data.Dataset,
                              validate_data: Optional[torch.utils.data.Dataset] = None,
                              validate_size: Optional[float] = None) -> Tuple[
                              bool, torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
        # whether split validation set

        validation = True
        if validate_data is None and validate_size is None:
            validation = False
            train_set, validate_set = train_data, None

        elif validate_data is None:
            validate_size = int(validate_size * len(train_data))
            train_size = len(train_data) - validate_size
            train_set, validate_set = torch.utils.data.random_split(
                train_data, [train_size, validate_size])
            if len(validate_set) == 0:
                validation = False
                validate_set = None

        else:
            train_set, validate_set = train_data, validate_data
        return validation, train_set, validate_set

    def fit(self, train_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int, validate_data: Optional[torch.utils.data.Dataset] = None,
            validate_size: Optional[float] = None, saved=False, save_path: Optional[str] = None):
        """training"""
        validation, train_set, validate_set = self._split_train_validate(train_data, validate_data, validate_size)

        # tqdm.write('----start training-----')
        print(f'training data size={len(train_set)}')
        if validation:
            print(f'validation data size={len(validate_set)}')

        # training for epochs
        print('----start training-----')
        for epoch in range(epochs):
            print(f'\n--epoch=[{epoch + 1}/{epochs}]--')
            training_start_time = time()
            training_loss = self._train_epoch(train_set, batch_size, epoch)
            training_end_time = time()
            print(f'time={training_end_time - training_start_time}s, '
                  f'train loss={training_loss}')
            if validation:
                validate_result = self._validate_epoch(validate_set,
                                                       batch_size)
                print(f'      validation result: {dict2str(validate_result)}')
            # tqdm.write(f'epoch={epoch}, '
            #            f'time={training_end_time - training_start_time}s, '
            #            f'train loss={training_loss}')
            # tqdm.write(f'validation result: {validate_result}')
            if self.scheduler is not None:
                self.scheduler.step()

        # save the model
        if saved:
            if save_path is None:
                save_dir = os.path.join(os.getcwd(), "save")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = f"{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pth"
                save_path = os.path.join(save_dir, file_name)
            torch.save(self.model.state_dict(), save_path)
            print(f'model is saved as {save_path}')
