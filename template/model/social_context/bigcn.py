import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from copy import copy

from torch_scatter import scatter_mean

from template.model.model import AbstractModel
"""
User Preference-aware Fake News Detection
paper: https://arxiv.org/abs/2104.12259
code: https://github.com/safe-graph/GNN-FakeNews

Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
paper: https://arxiv.org/abs/2001.06362
"""


class _RumorGCN(torch.nn.Module):
    def __init__(self,
                 feature_size: int,
                 hidden_size: int,
                 out_size: int,
                 dropout_ratio=0.5):
        super(_RumorGCN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.conv1 = GCNConv(feature_size, hidden_size)
        self.conv2 = GCNConv(hidden_size + feature_size, out_size)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor,
                root_index: Tensor):
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
        x = scatter_mean(x, batch, dim=0)

        return x


class BiGCN(AbstractModel):
    def __init__(self, feature_size: int, hidden_size: int, out_size: int):
        """
        Args:
            out_size: output size for graph convolution layer
        """
        super(BiGCN, self).__init__()
        self.TDRumorGCN = _RumorGCN(feature_size, hidden_size, out_size)
        self.BURumorGCN = _RumorGCN(feature_size, hidden_size, out_size)
        self.fc = torch.nn.Linear((out_size + hidden_size) * 2, 2)

    def forward(self, x: Tensor, edge_index: Tensor, bu_edge_index: Tensor,
                batch: Tensor, root_index: Tensor):
        td_x = self.TDRumorGCN(x, edge_index, batch, root_index)
        bu_x = self.BURumorGCN(x, bu_edge_index, batch, root_index)
        x = torch.cat((td_x, bu_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def calculate_loss(self, data) -> torch.Tensor:
        output = self.forward(data.x, data.edge_index, data.BU_edge_index,
                              data.batch, data.root_index)
        loss = F.nll_loss(output, data.y)
        return loss

    def predict(self, data_without_label) -> torch.Tensor:
        output = self.forward(data_without_label.x,
                              data_without_label.edge_index,
                              data_without_label.BU_edge_index,
                              data_without_label.batch,
                              data_without_label.root_index)
        return F.softmax(output, dim=1)
