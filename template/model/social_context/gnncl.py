from math import ceil
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from template.model.model import AbstractModel

"""
Graph neural networks with continual learning for fake news detection from social media
paper: https://arxiv.org/abs/2007.03316
code: https://github.com/safe-graph/GNN-FakeNews
"""


class _DenseGNN(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 normalize=False,
                 fc=True):
        super(_DenseGNN, self).__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if fc is True:
            self.fc = torch.nn.Linear(2 * hidden_channels + out_channels,
                                      out_channels)
        else:
            self.fc = None

    def bn(self, i: int, x: Tensor):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
        x = torch.cat([x1, x2, x3], dim=-1)
        if self.fc is not None:
            x = F.relu(self.fc(x))
        return x


class GNNCL(AbstractModel):
    def __init__(self, feature_size: int, max_nodes: int):
        """
        Args:
            feature_size (int): dimension of input node feature
            max_nodes (int): maximum number of nodes in a graph
        """
        super(GNNCL, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = _DenseGNN(feature_size, 64, num_nodes)
        self.gnn1_embed = _DenseGNN(feature_size, 64, 64, fc=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = _DenseGNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = _DenseGNN(3 * 64, 64, 64, fc=False)

        self.gnn3_embed = _DenseGNN(3 * 64, 64, 64, fc=False)

        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None):
        """

        Args:
            x (Tensor): node feature, shape (batch_size, max_nodes, feature_size)
            adj (Tensor): adjacency matrix, shape (batch_size, max_nodes, max_nodes)
            mask (Optional[Tensor], optional): mask for adjacency matrix, shape (batch_size, max_nodes). Default=None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                output (Tensor): prediction of being fake, shape (batch_size, 2).
                l (Tensor): loss of first pooling layer.
                e (Tensor): entropy of first pooling layer.
        """
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out, l1 + l2, e1 + e2

    def calculate_loss(self, data) -> torch.Tensor:
        output, _, _ = self.forward(data.x, data.adj, data.mask)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(output, data.y.view(-1))
        return loss

    def predict(self, data_without_label) -> torch.Tensor:
        output, _, _ = self.forward(data_without_label.x,
                                    data_without_label.adj,
                                    data_without_label.mask)
        return F.softmax(output, dim=1)
