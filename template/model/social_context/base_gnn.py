import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool

from template.model.model import AbstractModel


class _BaseGNN(AbstractModel):
    def __init__(self, feature_size: int, hidden_size: int, concat=False):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
            concat (bool): concat news embedding and graph embedding. Default=False
        """
        super(_BaseGNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.concat = concat

        if self.concat:
            self.fc0 = torch.nn.Linear(self.feature_size, self.hidden_size)
            self.fc1 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor,
                num_graphs: int):
        """

        Args:
            x (Tensor): node feature, shape=(num_nodes, feature_size)
            edge_index (Tensor): edge index, shape=(2, num_edges)
            batch (Tensor): index of graph each node belongs to, shape=(num_nodes,)
            num_graphs (int): number of graphs, a.k.a. batch_size

        Returns:
            output (Tensor): prediction of being fake, shape=(num_graphs, 2)
        """
        edge_attr = None
        raw_x = x
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = global_max_pool(x, batch)

        # whether concat news embedding and graph embedding
        if self.concat:
            news = torch.stack([
                raw_x[(batch == idx).nonzero().squeeze()[0]]
                for idx in range(num_graphs)
            ])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_loss(self, data) -> torch.Tensor:
        output = self.forward(data.x, data.edge_index, data.batch,
                              data.num_graphs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, data.y)
        return loss

    def predict(self, data_without_label) -> torch.Tensor:
        output = self.forward(data_without_label.x,
                              data_without_label.edge_index,
                              data_without_label.batch,
                              data_without_label.num_graphs)
        return F.softmax(output, dim=1)


class GCN(_BaseGNN):
    """
    Semi-Supervised Classification with Graph Convolutional Networks
    paper: https://arxiv.org/abs/1609.02907
    code: https://github.com/safe-graph/GNN-FakeNews
    """
    def __init__(self, feature_size: int, hidden_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
        """
        super().__init__(feature_size, hidden_size, False)
        self.conv = GCNConv(self.feature_size, self.hidden_size)


class SAGE(_BaseGNN):
    """
    Inductive Representation Learning on Large Graphs
    paper: https://arxiv.org/abs/1706.02216
    code: https://github.com/safe-graph/GNN-FakeNews
    """
    def __init__(self, feature_size: int, hidden_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
        """
        super().__init__(feature_size, hidden_size, False)
        self.conv = SAGEConv(self.feature_size, self.hidden_size)


class GAT(_BaseGNN):
    """
    Graph Attention Networks
    paper: https://arxiv.org/abs/1710.10903
    code: https://github.com/safe-graph/GNN-FakeNews
    """
    def __init__(self, feature_size: int, hidden_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
        """
        super().__init__(feature_size, hidden_size, False)
        self.conv = GATConv(self.feature_size, self.hidden_size)
