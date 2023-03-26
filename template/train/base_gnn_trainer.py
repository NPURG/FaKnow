import torch
from torch_geometric.loader import DataLoader

from template.train.trainer import BaseTrainer


class BaseGNNTrainer(BaseTrainer):
    def _train_epoch(self, loader: DataLoader, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()

        for batch_id, batch_data in enumerate(loader):
            loss = self.model.calculate_loss(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        outputs = []
        labels = []
        for batch_data in loader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data.y)
        return self.evaluator.evaluate(torch.concat(outputs),
                                       torch.concat(labels))
