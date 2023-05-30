import logging
import os
import sys
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
from faknow.utils.util import dict2str, seconds2str, now2str, check_loss_type, EarlyStopping


class AbstractTrainer:
    def __init__(self,
                 model: AbstractModel,
                 evaluator: Evaluator,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, data: DataLoader):
        """evaluate after training"""
        raise NotImplementedError

    def fit(self,
            train_loader: DataLoader,
            num_epochs: int,
            validate_loader: Optional[DataLoader] = None,
            save=False,
            save_path=None):
        raise NotImplementedError


class BaseTrainer(AbstractTrainer):
    def __init__(self,
                 model: AbstractModel,
                 evaluator: Evaluator,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device='cpu',
                 early_stopping: Optional[EarlyStopping] = None):
        super(BaseTrainer, self).__init__(model, evaluator, optimizer,
                                          scheduler, clip_grad_norm, device)

        # best validation score
        self.early_stopping = early_stopping
        self.best_score = 0.0
        self.best_epoch = 0

        # create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def _train_epoch(self, loader: DataLoader,
                     epoch: int) -> Union[float, Dict[str, float]]:
        """training for one epoch"""

        # switch model to train mode
        self.model.train()

        with tqdm(enumerate(loader),
                  total=len(loader),
                  ncols=100,
                  desc='Training') as pbar:
            loss = None
            result_is_dict = False
            for batch_id, batch_data in pbar:
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
        """validation after training for one epoch"""
        # evaluate
        result = self.evaluate(loader)

        # get the first metric as the validation score if accuracy is not in result
        if 'accuracy' in result:
            score = result['accuracy']
        else:
            if epoch == 0:
                warnings.warn(
                    'no accuracy in result, use the first metric as the validation score'
                )
            score = list(result.values())[0]
        return score, result

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
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

        result_file_name = f"{self.model.__class__.__name__}-{now2str()}"

        # 未指定保存路径时，取fit开始时刻作为文件名
        if save_best is not None and save_path is None:
            save_path = os.path.join(os.getcwd(), f"save/{result_file_name}.pth")

        # todo wjl 这一块logging的代码太臃肿了，优化然后提取出来作为一个函数
        # create files(tb_logs + logs)
        tb_logs_path = f"tb_logs/{result_file_name}"
        writer = SummaryWriter(tb_logs_path)

        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        file_name = f"{result_file_name}.log"
        logs_path = os.path.join(logs_dir, file_name)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logs_path)
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)

        self._show_data_size(train_loader, validate_loader)

        print(f'Tensorboard log is saved in {tb_logs_path}', file=sys.stderr)
        self.logger.info(f'Tensorboard log is saved in {tb_logs_path}')

        # training for num_epochs
        print('----start training-----', file=sys.stderr)
        for epoch in range(num_epochs):
            print(f'\n--epoch=[{epoch}/{num_epochs - 1}]--', file=sys.stderr)

            # todo wlj 这一块logging的代码太臃肿了，优化然后提取出来作为一个函数
            # create formatter and add it to the handlers
            formatter = logging.Formatter('')
            fh.setFormatter(formatter)

            # add the handlers to the logger
            self.logger.addHandler(fh)
            self.logger.info(f'\n--epoch=[{epoch}/{num_epochs - 1}]--')

            # create formatter and add it to the handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            # add the handlers to the logger
            self.logger.addHandler(fh)

            # train
            training_start_time = time()
            train_result = self._train_epoch(train_loader, epoch)

            # show training result
            cost_time_str = seconds2str(time() - training_start_time)
            self._show_train_result(train_result, cost_time_str, writer, epoch)

            # validate
            if validate_loader is not None:
                validation_score, validation_result = self._validate_epoch(
                    validate_loader, epoch)

                save_best = self._find_best_score(epoch, validation_score, save_best, save_path)

                self._show_validation_result(validation_result,
                                             validation_score,
                                             writer,
                                             epoch,
                                             save_best)

                if self.early_stopping is not None and self.early_stopping.early_stop:
                    break

            # learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        writer.flush()
        writer.close()

        # save last epoch model
        # do not use bool condition like `if not save_best`
        if save_best is False:
            self.save(save_path)

    def save(self, save_path: Optional[str] = None):

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
        print(f'\nmodel is saved in {save_path}', file=sys.stderr)

    def _find_best_score(self, epoch: int, validation_score: float, save_best: bool, save_path: str) -> bool:

        improvement = False
        if self.early_stopping is not None:
            save_best = True
            improvement = self.early_stopping(validation_score)
        elif save_best:
            improvement = validation_score > self.best_score

        # save best model
        if save_best and improvement:
            self.best_score = validation_score
            self.best_epoch = epoch
            self.save(save_path)

        return save_best

    def _move_data_to_device(self, batch_data) -> Any:
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

    def _show_train_result(self,
                           train_result: Union[float, Dict[str, float]],
                           cost_time_str: str,
                           writer,
                           epoch: int):

        self.logger.info(f'training time={cost_time_str}')
        print(f'training time={cost_time_str}', file=sys.stderr)

        # todo wjl 以后这里不要传入writer参数，而是直接调用self.writer
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
        else:
            raise TypeError(
                f"train_result type error: must be float or Dict[str, float], but got {type(train_result)}"
            )

    def _show_validation_result(self,
                                validation_result: Dict[str, float],
                                validation_score: float,
                                writer,
                                epoch: int,
                                save_best: Optional[bool] = None):
        # todo wjl 以后这里不要传入writer参数，而是直接调用self.writer
        for metric, value in validation_result.items():
            writer.add_scalar("Validation/" + metric, value, epoch)
        self.logger.info("validation result : " + dict2str(validation_result))
        print("validation result : " + dict2str(validation_result),
              file=sys.stderr)

        score_info = f"current score : {validation_score:.6f}"
        if save_best:
            score_info = score_info + f", best score : {self.best_score:.6f}, best epoch : {str(self.best_epoch)}"
        if self.early_stopping is not None and self.early_stopping.early_stop:
            score_info = score_info + f"\nearly stopping at epoch {epoch}!"
        self.logger.info(score_info)
        print(score_info, file=sys.stderr)

    def _show_data_size(self,
                        train_loader: DataLoader,
                        validate_loader: Optional[DataLoader] = None):
        train_set_size = len(train_loader.dataset)
        print(f'training data size={train_set_size}', file=sys.stderr)
        self.logger.info(f'training data size={train_set_size}')
        if validate_loader is not None:
            validate_set_size = len(validate_loader.dataset)
            print(f'validation data size={validate_set_size}', file=sys.stderr)
            self.logger.info(f'validation data size={validate_set_size}')
