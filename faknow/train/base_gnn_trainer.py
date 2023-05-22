import torch
from torch_geometric.loader import DataLoader

from faknow.train.trainer import BaseTrainer


class BaseGNNTrainer(BaseTrainer):

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
