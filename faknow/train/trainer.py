import datetime
import logging
import os
import sys
import warnings
from time import time
from typing import Optional, Dict, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from faknow.evaluate.evaluator import Evaluator
from faknow.model.model import AbstractModel
from faknow.utils.util import dict2str


class AbstractTrainer:
    def __init__(
            self,
            model: AbstractModel,
            evaluator: Evaluator,
            optimizer: Optimizer,
            scheduler: Optional[_LRScheduler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler

    @torch.no_grad()
    def evaluate(
            self,
            data: DataLoader
    ):
        """evaluate after training"""
        raise NotImplementedError

    def fit(
            self,
            train_data: DataLoader,
            num_epoch: int,
            validate_data=None,
            save=False,
            save_path=None
    ):
        raise NotImplementedError


class BaseTrainer(AbstractTrainer):
    def __init__(
            self,
            model: AbstractModel,
            evaluator: Evaluator,
            optimizer: Optimizer,
            scheduler: Optional[_LRScheduler] = None
    ):
        super(BaseTrainer, self).__init__(model, evaluator, optimizer, scheduler)

        # create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        """
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logs_path)
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        """

    @torch.no_grad()
    def evaluate(
            self,
            loader: DataLoader
    ):
        # evaluation mode
        self.model.eval()

        # start evaluating
        outputs = []
        labels = []
        for batch_data in loader:
            outputs.append(self.model.predict(batch_data))

            # todo 统一使用dict还是tuple
            # 是否要区分dict trainer和tuple trainer
            labels.append(batch_data['label'])
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(labels))

    def _train_epoch(
            self,
            loader: DataLoader,
            epoch: int
    ) -> Union[float, Dict[str, float]]:

        """training for one epoch"""

        # switch model to train mode
        self.model.train()

        with tqdm(enumerate(loader), total=len(loader), ncols=100, desc='Training') as pbar:
            loss = None
            result_is_dict = False
            for batch_id, batch_data in pbar:
                result = self.model.calculate_loss(batch_data)

                # check result type
                if type(result) is dict:
                    result_is_dict = True
                    if 'total_loss' in result.keys():
                        loss = result['total_loss']
                    else:
                        # todo 是否允许没有total_loss，采用所有loss的和作为total_loss
                        warnings.warn(f"no total_loss in result: {result}, use sum of all losses as total_loss")
                        loss = torch.sum(torch.stack(list(result.values())))
                elif type(result) is torch.Tensor:
                    loss = result
                else:
                    raise TypeError(f"result type error: {result}")

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if result_is_dict:
                return {k: v.item() for k, v in result.items()}
            return loss.item()

    def _validate_epoch(
            self,
            loader: DataLoader,
            epoch: int
    ) -> Dict[str, float]:
        """validation after training for one epoch"""
        # todo 计算best score，作为最佳结果保存
        # evaluate
        result = self.evaluate(loader)
        return result

    def save(
            self,
            save_path: str
    ):
        """save the model"""
        if save_path is None:
            save_dir = os.path.join(os.getcwd(), "save")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = f"{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pth"
            save_path = os.path.join(save_dir, file_name)

        torch.save(self.model.state_dict(), save_path)

        # save visualization(log + console)
        self.logger.info(f'\nmodel is saved as {save_path}')
        print(f'\nmodel is saved as {save_path}', file=sys.stderr)

    def fit(
            self,
            train_loader: DataLoader,
            num_epoch: int,
            validate_loader: Optional[DataLoader] = None,
            save=False,
            save_path: Optional[str] = None
    ):
        validation = True
        if validate_loader is None:
            validation = False

        # create files(tb_logs + logs)
        tb_logs_path = f"tb_logs/{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}"
        writer = SummaryWriter(tb_logs_path)

        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        file_name = f"{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.log"
        logs_path = os.path.join(logs_dir, file_name)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logs_path)
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)

        # print config information
        print(f'training data size={len(train_loader.dataset)}', file=sys.stderr)
        self.logger.info(f'training data size={len(train_loader.dataset)}')
        if validation:
            print(f'validation data size={len(validate_loader.dataset)}', file=sys.stderr)
            self.logger.info(f'validation data size={len(validate_loader.dataset)}')
        self.logger.info(f'Tensorboard log is saved as {tb_logs_path}')

        # training for num_epoch
        print('----start training-----', file=sys.stderr)
        for epoch in range(num_epoch):
            print(f'\n--epoch=[{epoch + 1}/{num_epoch}]--', file=sys.stderr)

            # create formatter and add it to the handlers
            formatter = logging.Formatter('')
            fh.setFormatter(formatter)

            # add the handlers to the logger
            self.logger.addHandler(fh)
            self.logger.info(f'\n--epoch=[{epoch + 1}/{num_epoch}]--')

            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            # add the handlers to the logger
            self.logger.addHandler(fh)

            # train
            training_start_time = time()
            train_result = self._train_epoch(train_loader, epoch)

            # show training time
            training_end_time = time()
            training_time = training_end_time - training_start_time

            # todo 把这一小段写成util函数，将时间输入，返回格式化后的字符串
            if training_time < 60:
                self.logger.info(f'training time={training_time:.1f}s')
                print(f'training time={training_time:.1f}s', file=sys.stderr)
            else:
                training_time = int(training_time)
                minutes = training_time // 60
                seconds = training_time % 60
                self.logger.info(f'training time={minutes}m{seconds:02d}s')
                print(f'training time={minutes}m{seconds:02d}s', file=sys.stderr)

            # show training loss
            if type(train_result) is float:
                # single loss
                writer.add_scalar("Train/loss", train_result, epoch)
                self.logger.info(f"training loss : loss={train_result:.6f}")
                print(f"training loss : loss={train_result:.6f}", file=sys.stderr)
            elif type(train_result) is dict:
                # multiple losses
                for metric, value in train_result.items():
                    writer.add_scalar("Train/" + metric, value, epoch)
                self.logger.info(f"training loss : {dict2str(train_result)}")
                print(f"training loss : {dict2str(train_result)}", file=sys.stderr)

            # validate
            if validation:
                validation_result = self._validate_epoch(validate_loader, epoch)

                # show validation result
                for metric, value in validation_result.items():
                    writer.add_scalar("Validation/" + metric, value, epoch)
                self.logger.info("validation result : " + dict2str(validation_result))
                print("validation result : " + dict2str(validation_result), file=sys.stderr)

            # learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        writer.flush()
        writer.close()

        # save the model
        if save:
            self.save(save_path)
