from copy import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

from template.model.model import AbstractModel

"""
Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
paper: https://arxiv.org/abs/2001.06362
code: https://github.com/safe-graph/GNN-FakeNews
"""


class _RumorGCN(torch.nn.Module):
    def __init__(self,
                 feature_size: int,
                 hidden_size: int,
                 out_size: int,
                 dropout_ratio=0.5):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer
            out_size (int): dimension of output layer
            dropout_ratio (float): drop out rate. Default=0.5
        """
        super(_RumorGCN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.conv1 = GCNConv(feature_size, hidden_size)
        self.conv2 = GCNConv(hidden_size + feature_size, out_size)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor,
                root_index: Tensor):
        """

        Args:
            x (Tensor): node feature, shape=(num_nodes, feature_size)
            edge_index (Tensor): edge index, shape=(2, num_edges)
            batch (Tensor): index of graph each node belongs to, shape=(num_nodes,)
            root_index (Tensor): index of root node for each graph, shape=(num_graphs,)

        Returns:
            out (Tensor): output of graph convolution layer, shape=(batch_size, 2 * out_size)
        """
        x1 = copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy(x)
        root_extend = torch.zeros(len(batch), x1.size(1)).to(root_index.device)
        batch_size = max(batch) + 1

        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(batch), x2.size(1)).to(root_index.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        out = scatter_mean(x, batch, dim=0)

        return out


class BiGCN(AbstractModel):
    """
    Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
    """
    def __init__(self, feature_size: int, hidden_size=128, out_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
            out_size (int): dimension of output layer. Default=128
        """
        super(BiGCN, self).__init__()
        # top-down
        self.TDRumorGCN = _RumorGCN(feature_size, hidden_size, out_size)
        # bottom-up
        self.BURumorGCN = _RumorGCN(feature_size, hidden_size, out_size)
        self.fc = torch.nn.Linear((out_size + hidden_size) * 2, 2)

    def forward(self, x: Tensor, edge_index: Tensor, bu_edge_index: Tensor,
                batch: Tensor, root_index: Tensor):
        """

        Args:
            x (Tensor): node feature, shape=(num_nodes, feature_size), shape=(num_nodes, feature_size)
            edge_index (Tensor): top-down edge index, shape=(2, num_edges)
            bu_edge_index (Tensor): bottom-up edge index, shape=(2, num_edges)
            batch (Tensor): index of graph each node belongs to, shape=(num_nodes,)
            root_index (Tensor): index of root node for each graph, shape=(num_graphs,)

        Returns:
            out (Tensor): prediction of being fake, shape=(batch_size, 2)
        """
        td_x = self.TDRumorGCN(x, edge_index, batch, root_index)
        bu_x = self.BURumorGCN(x, bu_edge_index, batch, root_index)
        x = torch.cat((td_x, bu_x), 1)
        out = self.fc(x)
        return out

    def calculate_loss(self, data) -> torch.Tensor:
        output = self.forward(data.x, data.edge_index, data.BU_edge_index,
                              data.batch, data.root_index)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, data.y)
        return loss

    def predict(self, data_without_label) -> torch.Tensor:
        output = self.forward(data_without_label.x,
                              data_without_label.edge_index,
                              data_without_label.BU_edge_index,
                              data_without_label.batch,
                              data_without_label.root_index)
        return torch.softmax(output, dim=1)
