from typing import Dict

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from faknow.train.trainer import BaseTrainer
from faknow.utils.pgd import PGD


class MFANTrainer(BaseTrainer):
    """
    Trainer for MFAN model with PGD defence,
    which inherits from BaseTrainer and modifies the '_train_epoch' method.
    """

    def _train_epoch(self, data: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()

        pgd_word = PGD(self.model,
                       emb_name='word_embedding',
                       epsilon=6,
                       alpha=1.8)
        losses = loss_defence = loss_adv = None

        with tqdm(enumerate(data),
                  total=len(data),
                  ncols=100,
                  desc='Training') as pbar:

            for batch_id, batch_data in pbar:
                # common loss
                losses = self.model.calculate_loss(batch_data)
                loss_defence = losses['total_loss']
                self.optimizer.zero_grad()
                loss_defence.backward()

                # PGD
                k = 3
                pgd_word.backup_grad()
                for t in range(k):
                    pgd_word.attack(is_first_attack=(t == 0))
                    if t != k - 1:
                        self.model.zero_grad()
                    else:
                        pgd_word.restore_grad()
                    y_pred = self.model.predict(batch_data)
                    loss_adv = F.cross_entropy(y_pred, batch_data['label'])
                    loss_adv.backward()
                pgd_word.restore()

                self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}
