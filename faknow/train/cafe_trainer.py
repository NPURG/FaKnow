from faknow.train.trainer import BaseTrainer
from typing import Optional, Dict, Any
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.cafe import CAFE
from faknow.utils.util import EarlyStopping


class CafeTrainer(BaseTrainer):
    """
    Trainer for CAFE model with tow model,
    which inherits from BaseTrainer and modifies the '_train_epoch' method.
    """
    def __init__(self,
                 model: CAFE,
                 evaluator: Evaluator,
                 detection_optimizer: Optimizer,
                 similarity_optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device='cpu',
                 early_stopping: Optional[EarlyStopping] = None):
        """
        Args:
            model (CAFE): the first faknow abstract model to train
            evaluator (Evaluator):  faknow evaluator for evaluation
            detection_optimizer (Optimizer): pytorch optimizer for training of the detection model
            similarity_optimizer (Optimizer): pytorch optimizer for training of the similarity model
            scheduler (_LRScheduler): learning rate scheduler. Defaults=None.
            clip_grad_norm (Dict[str, Any]): key args for
                torch.nn.utils.clip_grad_norm_. Defaults=None.
            device (str): device to use. Defaults='cuda:0'.
            early_stopping (EarlyStopping): early stopping for training.
                If None, no early stopping will be performed. Defaults=None.
        """

        super().__init__(model, evaluator, detection_optimizer, scheduler,
                         clip_grad_norm, device, early_stopping)
        self.similarity_optimizer = similarity_optimizer

    def _train_epoch(self, loader: DataLoader,
                     epoch: int) -> Dict[str, float]:
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
        self.model.train()
        with tqdm(enumerate(loader),
                  total=len(loader),
                  ncols=100,
                  desc='Training') as pbar:
            loss = None
            for batch_id, batch_data in pbar:
                batch_data = self._move_data_to_device(batch_data)

                similarity_loss = self.model.similarity_module.calculate_loss(
                    batch_data)
                self.similarity_optimizer.zero_grad()
                similarity_loss.backward()
                self.similarity_optimizer.step()

                detection_loss = self.model.calculate_loss(batch_data)
                self.optimizer.zero_grad()
                detection_loss.backward()
                self.optimizer.step()

                loss = {
                    'similarity_loss': similarity_loss.item(),
                    'detection_loss': detection_loss.item()
                }

            return loss
