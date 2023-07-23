import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch

from faknow.model.model import AbstractModel


class _BaseGCNFN(AbstractModel):
    """
    base GCNFN model for GCNFN
    """
    def __init__(self,
                 feature_size: int,
                 hidden_size=128,
                 dropout_ratio=0.5,
                 concat=False):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
            dropout_ratio (float): dropout ratio. Default=0.5
            concat (bool): concat news embedding and graph embedding. Default=False
        """
        super(_BaseGCNFN, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.concat = concat

        self.conv1 = GATConv(self.feature_size, self.hidden_size * 2)
        self.conv2 = GATConv(self.hidden_size * 2, self.hidden_size * 2)

        if self.concat:
            self.fc0 = nn.Linear(self.feature_size, self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 2)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor,
                num_graphs: int):
        """
        Args:
            x (Tensor): node feature, shape=(num_nodes, feature_size)
            edge_index (Tensor): edge index, shape=(2, num_edges)
            batch (Tensor): index of graph each node belongs to, shape=(num_nodes,)
            num_graphs (int): number of graphs, a.k.a. batch_size

        Returns:
            Tensor: prediction of being fake, shape=(num_graphs, 2)
        """

        raw_x = x
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # whether concat news embedding and graph embedding
        if self.concat:
            news = torch.stack([
                raw_x[(batch == idx).nonzero().squeeze()[0]]
                for idx in range(num_graphs)
            ])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

    def calculate_loss(self, data: Batch) -> torch.Tensor:
        """
        calculate loss via CrossEntropyLoss

        Args:
            data (Batch): batch data

        Returns:
            torch.Tensor: loss
        """

        output = self.forward(data.x, data.edge_index, data.batch,
                              data.num_graphs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, data.y)
        return loss

    def predict(self, data_without_label: Batch) -> torch.Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Batch): batch data

        Returns:
            Tensor: softmax probability, shape=(num_graphs, 2)
        """

        output = self.forward(data_without_label.x,
                              data_without_label.edge_index,
                              data_without_label.batch,
                              data_without_label.num_graphs)
        return F.softmax(output, dim=1)


class GCNFN(_BaseGCNFN):
    """
    Fake news detection on social media using geometric deep learning, arXiv 2019
    paper: https://arxiv.org/abs/1902.06673
    code: https://github.com/safe-graph/GNN-FakeNews
    """

    def __init__(self, feature_size: int, hidden_size=128, dropout_ratio=0.5):
        """
        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
            dropout_ratio (float): dropout ratio. Default=0.5
        """

        super(GCNFN, self).__init__(feature_size, hidden_size, dropout_ratio,
                                    False)
