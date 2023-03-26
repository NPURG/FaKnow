import torch
from torch_geometric.loader import DenseDataLoader

from template.train.base_gnn_trainer import BaseGNNTrainer


class DenseGNNTrainer(BaseGNNTrainer):
    def _train_epoch(self, loader: DenseDataLoader, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()

        for batch_id, batch_data in enumerate(loader):
            loss = self.model.calculate_loss(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    @torch.no_grad()
    def evaluate(self, loader: DenseDataLoader):
        self.model.eval()
        outputs = []
        labels = []
        for batch_data in loader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data.y.view(-1))
        return self.evaluator.evaluate(torch.concat(outputs),
                                       torch.concat(labels))
