import torch
from torch_geometric.loader import DenseDataLoader

from faknow.train.base_gnn_trainer import BaseGNNTrainer


class DenseGNNTrainer(BaseGNNTrainer):
    """
    Base trainer for GNN models with dense batch data,
    which inherits from BaseGNNTrainer and modifies the evaluate method.
    """

    @torch.no_grad()
    def evaluate(self, loader: DenseDataLoader):
        """
        Evaluate model performance on testing or validation data.

        Args:
            loader (DenseDataLoader): pyg dense data to evaluate,
                where each batch data is torch_geometric.data.Batch
                with all attributes stacked in a new dimension.

        Returns:
            Dict[str, float]: evaluation metrics
        """

        self.model.eval()
        outputs = []
        labels = []
        for batch_data in loader:
            outputs.append(self.model.predict(batch_data))
            labels.append(batch_data.y.view(-1))
        return self.evaluator.evaluate(torch.concat(outputs),
                                       torch.concat(labels))
