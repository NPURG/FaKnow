import torch
from torch.utils.data.dataloader import DataLoader

from template.train.trainer import BaseTrainer


class EANNTrainer(BaseTrainer):

    def _train_epoch(self, data, batch_size: int, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        dataloader = DataLoader(data, batch_size, shuffle=True)

        # todo optimizer变化
        p = float(epoch) / 100
        lr = 0.001 / (1. + 10 * p) ** 0.75
        self.optimizer.lr = lr

        msg = None
        for batch_id, batch_data in enumerate(dataloader):
            # using trainer.loss_func first or model.calculate_loss
            if self.loss_func is None:
                loss, msg = self.model.calculate_loss(batch_data)
            else:
                loss = self.loss_func(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if msg is not None:
            print(msg)
        return loss
