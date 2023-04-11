import torch
from torch_geometric.loader import DenseDataLoader

from faknow.train.base_gnn_trainer import BaseGNNTrainer


class DenseGNNTrainer(BaseGNNTrainer):

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
