import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from faknow.train.trainer import BaseTrainer


class MCANTrainer(BaseTrainer):
    def _train_epoch(
            self,
            loader: DataLoader,
            epoch: int
    ) -> float:
        self.model.train()

        pbar = tqdm(enumerate(loader), total=len(loader), ncols=100, desc='Training')

        loss = others = None
        for batch_id, batch_data in pbar:
            loss = self.model.calculate_loss(batch_data)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=1.0)
            self.optimizer.step()

            pbar.set_postfix_str(f"loss={loss.item()}")

        pbar.close()

        return loss.item()
