import datetime
import os
from time import time
from typing import Optional, Callable

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
    def evaluate(self, data: DataLoader, batch_size: int):
        """evaluate after training"""
        raise NotImplementedError

    def fit(self, train_data: DataLoader,
            num_epoch: int, validate_data=None,
            save=False, save_path=None):
        raise NotImplementedError


class BaseTrainer(AbstractTrainer):
    def __init__(self, model: AbstractModel, evaluator: Evaluator,
                 optimizer: Optimizer, scheduler: Optional[_LRScheduler] = None, loss_func: Optional[Callable] = None):
        super(BaseTrainer, self).__init__(model, evaluator, optimizer, scheduler, loss_func)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        outputs = []
        labels = []
        for batch_data in loader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data['label'])
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(labels))

    def _train_epoch(self, loader: DataLoader, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        # todo tqdm与print冲突
        # data_iter = tenumerate(
        #     dataloader,
        #     total=len(dataloader),
        #     ncols=100,
        #     desc=f'Training',
        #     leave=False
        # )
        msg = loss = None
        for batch_id, batch_data in enumerate(loader):
            # using trainer.loss_func first or model.calculate_loss
            if self.loss_func is None:
                result = self.model.calculate_loss(batch_data)
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

    def _validate_epoch(self, loader: DataLoader):
        """validation after training for one epoch"""
        # todo 计算best score，作为最佳结果保存
        return self.evaluate(loader)

    # def _split_train_validate(self, train_data: torch.utils.data.Dataset,
    #                           validate_data: Optional[torch.utils.data.Dataset] = None,
    #                           validate_size: Optional[float] = None) -> Tuple[
    #                           bool, torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
    #     # whether split validation set
    #
    #     validation = True
    #     if validate_data is None and validate_size is None:
    #         validation = False
    #         train_set, validate_set = train_data, None
    #
    #     elif validate_data is None:
    #         validate_size = int(validate_size * len(train_data))
    #         train_size = len(train_data) - validate_size
    #         train_set, validate_set = torch.utils.data.random_split(
    #             train_data, [train_size, validate_size])
    #         if len(validate_set) == 0:
    #             validation = False
    #             validate_set = None
    #
    #     else:
    #         train_set, validate_set = train_data, validate_data
    #     return validation, train_set, validate_set

    def save(self, save_path: str):
        """save the model"""

        if save_path is None:
            save_dir = os.path.join(os.getcwd(), "save")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = f"{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pth"
            save_path = os.path.join(save_dir, file_name)

        torch.save(self.model.state_dict(), save_path)
        print(f'model is saved as {save_path}')

    def fit(self, train_loader: DataLoader,
            num_epoch: int, validate_loader: Optional[DataLoader] = None,
            save=False, save_path: Optional[str] = None):
        """training"""
        validation = True
        if validate_loader is None:
            validation = False
        # validation, train_set, validate_set = self._split_train_validate(train_data, validate_data, validate_size)

        # tqdm.write('----start training-----')
        print(f'training data size={len(train_loader.dataset)}')
        if validation:
            print(f'validation data size={len(validate_loader.dataset)}')

        # training for num_epoch
        print('----start training-----')
        for epoch in range(num_epoch):
            print(f'\n--epoch=[{epoch + 1}/{num_epoch}]--')
            training_start_time = time()
            training_loss = self._train_epoch(train_loader, epoch)
            training_end_time = time()
            print(f'time={training_end_time - training_start_time}s, '
                  f'train loss={training_loss}')
            if validation:
                validate_result = self._validate_epoch(validate_loader)
                print(f'      validation result: {dict2str(validate_result)}')
            # tqdm.write(f'epoch={epoch}, '
            #            f'time={training_end_time - training_start_time}s, '
            #            f'train loss={training_loss}')
            # tqdm.write(f'validation result: {validate_result}')
            if self.scheduler is not None:
                self.scheduler.step()

        # save the model
        if save:
            self.save(save_path)
