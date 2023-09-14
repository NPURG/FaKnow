import torch
from torch_geometric.loader import DataLoader

from faknow.train.trainer import BaseTrainer


class BaseGNNTrainer(BaseTrainer):
    """
    Base trainer for GNN models,
    which inherits from BaseTrainer and modifies the evaluate method.
    """

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """
        Evaluate model performance on testing or validation data.

        Args:
            loader (DataLoader): pyg data to evaluate,
                where each batch data is torch_geometric.data.Batch
                and each sample data in a batch is torch_geometric.data.Data

        Returns:
            Dict[str, float]: evaluation metrics
        """

        self.model.eval()
        outputs = []
        labels = []
        for batch_data in loader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data.y)
        return self.evaluator.evaluate(torch.concat(outputs),
                                       torch.concat(labels))
