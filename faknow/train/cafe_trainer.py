import torch
from faknow.train.trainer import BaseTrainer
import logging
import os
from time import time
from typing import Optional, Dict, Union, Any
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from faknow.evaluate.evaluator import Evaluator
from faknow.model.model import AbstractModel
from faknow.utils.util import (dict2str, seconds2str, now2str, check_loss_type,
                               EarlyStopping)


class CafeTrainer(BaseTrainer):
    """
    Trainer for CAFE model with tow model,
    which inherits from BaseTrainer and modifies the '_train_epoch' method.
    """

    def __init__(self,
                 model_1: AbstractModel,
                 model_2: AbstractModel,
                 evaluator_1: Evaluator,
                 evaluator_2: Evaluator,
                 optimizer_1: Optimizer,
                 optimizer_2: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device='cuda:0',
                 early_stopping: Optional[EarlyStopping] = None):
        """
        Args:
            model_1 (AbstractModel): the first faknow abstract model to train
            model_2 (AbstractModel): the second faknow abstract model to train
            evaluator_1 (Evaluator):  faknow evaluator for evaluation of the first model
            evaluator_2 (Evaluator):  faknow evaluator for evaluation of the second model
            optimizer_1 (Optimizer): pytorch optimizer for training of the first model
            optimizer_2 (Optimizer): pytorch optimizer for training of the second model
            scheduler (_LRScheduler): learning rate scheduler. Defaults=None.
            clip_grad_norm (Dict[str, Any]): key args for
                torch.nn.utils.clip_grad_norm_. Defaults=None.
            device (str): device to use. Defaults='cuda:0'.
            early_stopping (EarlyStopping): early stopping for training.
                If None, no early stopping will be performed. Defaults=None.
        """

        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1.evaluator = evaluator_1
        self.model_2.evaluator = evaluator_2
        self.model_1.optimizer = optimizer_1
        self.model_2.optimizer = optimizer_2
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        # best validation score
        self.early_stopping = early_stopping
        self.best_score = 0.0
        self.best_epoch = 0
        self.device = device
        self.writer = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def fit(self,
            train_loader: DataLoader,
            num_epochs: int,
            validate_loader: Optional[DataLoader] = None,
            save_best: Optional[bool] = None,
            save_path: Optional[str] = None):
        """
        Fit model_1 and model_2, including training, validation and save model.
        Results will be logged in console, log file and tensorboard.
        Args:
            train_loader (DataLoader): training data
            num_epochs (int): number of epochs to train
            validate_loader (DataLoader): validation data.
                If None, no validation. Defaults=None.
            save_best (bool): whether to save model with best validation score.
                If False, save the last epoch model.
                If None, do not save any model.
                Defaults=None.
            save_path (str): path to save model, if save_best is not None.
                If None, save in './save/model_name-current_time.pth'.
                Defaults=None.
        """
        result_file_name = f"{self.model_1.__class__.__name__}-{self.model_2.__class__.__name__}-{now2str()}"

        # 未指定保存路径时，取fit开始时刻作为文件名
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     f"save/{result_file_name}.pth")

        # log
        tb_logs_path = f"tb_logs/{result_file_name}"
        self.writer = SummaryWriter(tb_logs_path)
        self._BaseTrainer__add_file_log(result_file_name)

        self.logger.info(f'Tensorboard log is saved in {tb_logs_path}')
        self.logger.info(f'log file is saved in logs/{tb_logs_path}.log\n')
        self._show_data_size(train_loader, validate_loader)
        self.logger.info('----start training-----\n')

        for epoch in range(num_epochs):
            self.logger.info(f'epoch=[{epoch}/{num_epochs - 1}]')

            # train
            training_start_time = time()
            self.model_1.train_result = self._train_epoch(self.model_1, train_loader, epoch)
            self.model_2.train_result = self._train_epoch(self.model_2, train_loader, epoch)

            # show training result
            cost_time_str = seconds2str(time() - training_start_time)
            for model in [self.model_1, self.model_2]:
                self._show_train_result(model.train_result, cost_time_str, epoch)

            # validate
            if validate_loader is not None:
                validation_score, validation_result = self._validate_epoch(
                    validate_loader, epoch)

                save_best = self._find_best_score(epoch, validation_score,
                                                  save_best, save_path)

                self._show_validation_result(validation_result,
                                             validation_score, epoch,
                                             save_best)

                if self.early_stopping is not None and \
                        self.early_stopping.early_stop:
                    break

            # learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        self.writer.flush()
        self.writer.close()

        # save last epoch model
        # do not use bool condition like `if not save_best`
        if save_best is False:
            self.save(save_path)

    def _train_epoch(self, model: AbstractModel, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
         training for one epoch, including gradient clipping

         Args:
             model (AbstractModel): training model
             loader (DataLoader): training data
             epoch (int): current epoch

         Returns:
             Union[float, Dict[str, float]]: loss of current epoch.
                 If multiple losses,
                 return a dict of losses with loss name as key.
         """
        model.train()
        with tqdm(enumerate(loader),
                  total=len(loader),
                  ncols=100,
                  desc='Training') as pbar:
            loss = None
            result_is_dict = False
            for batch_id, batch_data in pbar:
                batch_data = self._move_data_to_device(batch_data)
                result = model.calculate_loss(batch_data)

                # check result type
                loss, result_is_dict = check_loss_type(result)

                # backward
                model.optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        **self.clip_grad_norm)
                model.optimizer.step()

            if result_is_dict:
                return {k: v.item() for k, v in result.items()}
            return loss.item()

        return {k: v.item() for k, v in losses.items()}

    def evaluate(self, loader: DataLoader):
        """
        Evaluate model performance on testing or validation data.

        Args:
            loader (DataLoader): data to evaluate,
                where each batch data is a dict with key 'label'

        Returns:
            Dict[str, float]: evaluation metrics
        """

        self.model_2.eval()

        outputs = []
        labels = []
        for batch_data in loader:
            batch_data = self._move_data_to_device(batch_data)
            outputs.append(self.model_2.predict(batch_data))

            # todo 统一使用dict还是tuple 是否要区分dict trainer和tuple trainer
            labels.append(batch_data[-1])
        return self.model_2.evaluator.evaluate(torch.concat(outputs),
                                               torch.concat(labels))
