import logging
import os
import warnings
from time import time
from typing import Optional, Dict, Union, Any, Tuple

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from faknow.evaluate.evaluator import Evaluator
from faknow.model.model import AbstractModel
from faknow.utils.util import (dict2str, seconds2str, now2str, check_loss_type,
                               EarlyStopping)


class AbstractTrainer:
    """
    Abstract trainer for FaKnow,
    including training, validation, evaluation and save model.
    All trainers should inherit from this class.

    Attributes:
        model (AbstractModel): faknow abstract model to train
        evaluator (Evaluator): faknow evaluator for evaluation
        optimizer (Optimizer): pytorch optimizer for training
        scheduler (_LRScheduler): learning rate scheduler.
        clip_grad_norm (Dict[str, Any]): key args for
            torch.nn.utils.clip_grad_norm_.
        device (torch.device): device to use.
    """
    def __init__(self,
                 model: AbstractModel,
                 evaluator: Evaluator,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device='cpu'):
        """
        Args:
            model (AbstractModel): faknow abstract model to train
            evaluator (Evaluator): faknow evaluator for evaluation
            optimizer (Optimizer): pytorch optimizer for training
            scheduler (_LRScheduler): learning rate scheduler. Defaults=None.
            clip_grad_norm (Dict[str, Any]): key args for
                torch.nn.utils.clip_grad_norm_. Defaults=None.
            device (str): device to use. Defaults='cpu'.
        """

        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """
        evaluate after training

        Args:
            loader (DataLoader): data to evaluate
        """

        raise NotImplementedError

    def fit(self,
            train_loader: DataLoader,
            num_epochs: int,
            validate_loader: Optional[DataLoader] = None,
            save=False,
            save_path=None):
        """
        train model

        Args:
            train_loader (DataLoader): training data
            num_epochs (int): number of epochs to train
            validate_loader (DataLoader): validation data.
                If None, no validation. Defaults=None.
            save (bool): whether to save model. Defaults=False.
            save_path (str): path to save model, if save=True.
                Defaults=None.
        """

        raise NotImplementedError

    def to(self, device: str, **kwargs):
        """
        move model and data to device

        Args:
            device (str): device to use
            **kwargs: other args for nn.Module.to
        """

        self.device = torch.device(device)
        self.model.to(self.device, **kwargs)

    def cuda(self, device='cuda:0', **kwargs):
        """
        move model and data to cuda

        Args:
            device (str): cuda device to use. Defaults='cuda:0'.
            **kwargs: other args for nn.Module.to
        """

        self.to(device, **kwargs)

    def cpu(self, **kwargs):
        """
        move model and data to cpu

        Args:
            **kwargs: other args for nn.Module.to
        """

        self.to('cpu', **kwargs)

    def _move_data_to_device(self, batch_data) -> Any:
        """
        move input batch data to device

        Args:
            batch_data (Any): input batch data

        Returns:
            Any: batch data on device
        """

        if type(batch_data) is dict:
            for k, v in batch_data.items():
                if type(v) is torch.Tensor:
                    batch_data[k] = v.to(self.device)
                else:
                    # todo 递归字典的情况
                    batch_data[k] = self._move_data_to_device(v)
        elif type(batch_data) is tuple:
            batch_data = tuple(value.to(self.device) for value in batch_data)
        else:
            batch_data = batch_data.to(self.device)
        return batch_data


class BaseTrainer(AbstractTrainer):
    """
    Base trainer for FaKnow, which inherits from AbstractTrainer
    and can be applied to most common tasks.

    Attributes:
        model (AbstractModel): faknow abstract model to train
        evaluator (Evaluator): faknow evaluator for evaluation
        optimizer (Optimizer): pytorch optimizer for training
        scheduler (_LRScheduler): learning rate scheduler.
        clip_grad_norm (Dict[str, Any]): key args for
            torch.nn.utils.clip_grad_norm_.
        device (torch.device): device to use.
        early_stopping (EarlyStopping): early stopping for training.
        best_score (float): best validation score for saving best model.
        best_epoch (int): best epoch for saving best model.
        writer (SummaryWriter): tensorboard writer.
        logger (logging.Logger): logger for logging in console and file.
    """
    def __init__(self,
                 model: AbstractModel,
                 evaluator: Evaluator,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device='cpu',
                 early_stopping: Optional[EarlyStopping] = None):
        """
        Args:
            model (AbstractModel): faknow abstract model to train
            evaluator (Evaluator): faknow evaluator for evaluation
            optimizer (Optimizer): pytorch optimizer for training
            scheduler (_LRScheduler): learning rate scheduler. Defaults=None.
            clip_grad_norm (Dict[str, Any]): key args for
                torch.nn.utils.clip_grad_norm_. Defaults=None.
            device (str): device to use. Defaults='cpu'.
            early_stopping (EarlyStopping): early stopping for training.
                If None, no early stopping will be performed. Defaults=None.
        """

        super(BaseTrainer, self).__init__(model, evaluator, optimizer,
                                          scheduler, clip_grad_norm, device)

        # best validation score
        self.early_stopping = early_stopping
        self.best_score = 0.0
        self.best_epoch = 0

        self.writer = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def _train_epoch(self, loader: DataLoader,
                     epoch: int) -> Union[float, Dict[str, float]]:
        """
        training for one epoch, including gradient clipping

        Args:
            loader (DataLoader): training data
            epoch (int): current epoch

        Returns:
            Union[float, Dict[str, float]]: loss of current epoch.
                If multiple losses,
                return a dict of losses with loss name as key.
        """

        # switch model to train mode
        self.model.train()

        with tqdm(enumerate(loader),
                  total=len(loader),
                  ncols=100,
                  desc='Training') as pbar:
            loss = None
            result_is_dict = False
            for batch_id, batch_data in pbar:
                batch_data = self._move_data_to_device(batch_data)
                result = self.model.calculate_loss(batch_data)

                # check result type
                loss, result_is_dict = check_loss_type(result)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        **self.clip_grad_norm)
                self.optimizer.step()

            if result_is_dict:
                return {k: v.item() for k, v in result.items()}
            return loss.item()

    def _validate_epoch(self, loader: DataLoader,
                        epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate after training for one epoch and return evaluation metrics.
        The accuracy will be taken as the validation score
        if it is included in the evaluation metrics.
        Otherwise, the first metric will be taken as default.

        Args:
            loader (DataLoader): validation data
            epoch (int): current epoch

        Returns:
            tuple:
                score (float): validation score
                result (Dict[str, float]): evaluation metrics
        """

        result = self.evaluate(loader)
        warnings.warn('no accuracy in result, use the first metric \
                as the validation score')
        score = list(result.values())[0]
        return score, result

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """
        Evaluate model performance on testing or validation data.

        Args:
            loader (DataLoader): data to evaluate,
                where each batch data is a dict with key 'label'

        Returns:
            Dict[str, float]: evaluation metrics
        """

        self.model.eval()

        outputs = []
        labels = []
        for batch_data in loader:
            batch_data = self._move_data_to_device(batch_data)
            outputs.append(self.model.predict(batch_data))

            # todo 统一使用dict还是tuple 是否要区分dict trainer和tuple trainer
            labels.append(batch_data['label'])
        return self.evaluator.evaluate(torch.concat(outputs),
                                       torch.concat(labels))

    def fit(self,
            train_loader: DataLoader,
            num_epochs: int,
            validate_loader: Optional[DataLoader] = None,
            save_best: Optional[bool] = None,
            save_path: Optional[str] = None):
        """
        Fit the model, including training, validation and save model.
        Results will be logged in console, log file and tensorboard.
        Early stopping has higher priority than save_best if it is not None.

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

        result_file_name = f"{self.model.__class__.__name__}-{now2str()}"

        # 未指定保存路径时，取fit开始时刻作为文件名
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     f"save/{result_file_name}.pth")

        # log
        tb_logs_path = f"tb_logs/{result_file_name}"
        self.writer = SummaryWriter(tb_logs_path)
        self.__add_file_log(result_file_name)

        self.logger.info(f'Tensorboard log is saved in {tb_logs_path}')
        self.logger.info(f'log file is saved in logs/{tb_logs_path}.log\n')
        self._show_data_size(train_loader, validate_loader)
        self.logger.info('----start training-----\n')

        for epoch in range(num_epochs):
            self.logger.info(f'epoch=[{epoch}/{num_epochs - 1}]')

            # train
            training_start_time = time()
            train_result = self._train_epoch(train_loader, epoch)

            # show training result
            cost_time_str = seconds2str(time() - training_start_time)
            self._show_train_result(train_result, cost_time_str, epoch)

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
        elif save_best is True:
            self.logger.info(f"load best model in epoch {self.best_epoch}")
            self.model.load_state_dict(torch.load(save_path))

    def save(self, save_path: Optional[str] = None):
        """
        save the model

        Args:
            save_path (str): path to save model.
                If None, save in './save/model_name-current_time.pth'.
                Defaults=None.
        """

        # default save path: './save/model_name-current_time.pth'
        if save_path is None:
            save_dir = os.path.join(os.getcwd(), "save")
            file_name = f"{self.model.__class__.__name__}-{now2str()}.pth"
            save_path = os.path.join(save_dir, file_name)

        # create save dir if not exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        torch.save(self.model.state_dict(), save_path)

        self.logger.info(f'\nmodel is saved in {save_path}')

    def _find_best_score(self, epoch: int, validation_score: float,
                         save_best: bool, save_path: str) -> bool:
        """
        Whether the current validation score is the best score,
        and save the model if so.

        Args:
            epoch (int): current epoch
            validation_score (float): current validation score
            save_best (bool): whether to save model with the best validation score.
                If False, save the last epoch model.
            save_path (str): path to save model, if save_best is not None.

        Returns:
            bool: whether to save model with the best validation score.
        """

        improvement = False
        if self.early_stopping is not None:
            # early stopping
            save_best = True
            improvement = self.early_stopping(validation_score)
        elif save_best:
            # validation score improved
            improvement = validation_score > self.best_score

        # save best model if save_best is True and validation score improved
        if save_best and improvement:
            self.best_score = validation_score
            self.best_epoch = epoch
            self.save(save_path)

        return save_best

    def _show_train_result(self, train_result: Union[float, Dict[str, float]],
                           cost_time_str: str, epoch: int):
        """
        Show training results in logging and tensorboard.

        Args:
            train_result (Union[float, Dict[str, float]]): training loss.
                If multiple losses,
                return a dict of losses with loss name as key.
            cost_time_str (str): training time in string format
            epoch (int): current epoch

        Raises:
            TypeError: train_result type error:
                must be float or Dict[str, float]
        """

        self.logger.info(f'training time={cost_time_str}')

        if type(train_result) is float:
            # single loss
            self.writer.add_scalar("Train/loss", train_result, epoch)
            self.logger.info(f"training loss : loss={train_result:.6f}")
        elif type(train_result) is dict:
            # multiple losses
            for metric, value in train_result.items():
                self.writer.add_scalar("Train/" + metric, value, epoch)
            self.logger.info(f"training loss : {dict2str(train_result)}")
        else:
            raise TypeError(
                f"train_result type error: must be float or Dict[str, float],\
                    but got {type(train_result)}"
            )

    def _show_validation_result(self,
                                validation_result: Dict[str, float],
                                validation_score: float,
                                epoch: int,
                                save_best: Optional[bool] = None):
        """
        Show validation results in logging and tensorboard.
        If save_best=True or self.early_stopping is not None,
        show the best validation score and epoch.

        Args:
            validation_result (Dict[str, float]): evaluate metrics
            validation_score (float): validation score
            epoch (int): current epoch
            save_best (bool): whether to save model with the best validation score.
                Defaults=None.
        """

        for metric, value in validation_result.items():
            self.writer.add_scalar("Validation/" + metric, value, epoch)
        self.logger.info("validation result : " + dict2str(validation_result))

        score_info = f"current score : {validation_score:.6f}"
        if save_best:
            score_info = score_info + f", best score : {self.best_score:.6f},\
                best epoch : {str(self.best_epoch)}"
        if self.early_stopping is not None and self.early_stopping.early_stop:
            score_info = score_info + f"\nearly stopping at epoch {epoch}!"

        self.logger.info(score_info)

    def _show_data_size(self,
                        train_loader: DataLoader,
                        validate_loader: Optional[DataLoader] = None):
        """
        show training data size and validation data size

        Args:
            train_loader (DataLoader): training data
            validate_loader (DataLoader): validation data. Defaults=None.
        """

        train_set_size = len(train_loader.dataset)
        self.logger.info(f'training data size={train_set_size}')
        if validate_loader is not None:
            validate_set_size = len(validate_loader.dataset)
            self.logger.info(f'validation data size={validate_set_size}')

    def __add_file_log(self, file_name: str):
        """
        add file log handler to logger

        Args:
            file_name (str): file name of log file
        """

        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_file_path = os.path.join(logs_dir, f"{file_name}.log")

        fh = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
