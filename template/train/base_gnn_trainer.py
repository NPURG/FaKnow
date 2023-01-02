import torch
from torch_geometric.loader import DataLoader

from template.train.trainer import BaseTrainer


class BaseGNNTrainer(BaseTrainer):
    def _train_epoch(self, data, batch_size: int, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        dataloader = DataLoader(data, batch_size, shuffle=True)

        for batch_id, batch_data in enumerate(dataloader):
            # using trainer.loss_func first or model.calculate_loss
            if self.loss_func is None:
                loss = self.model.calculate_loss(batch_data)
            else:
                loss = self.loss_func(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    @torch.no_grad()
    def evaluate(self, data, batch_size: int):
        self.model.eval()
        dataloader = DataLoader(data, batch_size, shuffle=False)
        outputs = []
        labels = []
        for batch_data in dataloader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data.y)
        return self.evaluator.evaluate(torch.concat(outputs),
                                       torch.concat(labels))
