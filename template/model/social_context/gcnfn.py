from template.model.model import AbstractModel
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
"""
using two GCN layers and one mean-pooling layer
Vanilla GCNFN: concat = False, feature = content
UPFD-GCNFN: concat = True, feature = spacy
"""


class GCNFN(AbstractModel):
    def __init__(self,
                 feature_size: int,
                 hidden_size: int,
                 dropout_ratio=0.5,
                 concat=False):
        super(GCNFN, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.concat = concat
        # todo 使用的是GAT而非GCN
        self.conv1 = GATConv(self.feature_size, self.hidden_size * 2)
        self.conv2 = GATConv(self.hidden_size * 2, self.hidden_size * 2)

        if self.concat:
            self.fc0 = nn.Linear(self.feature_size, self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # whether concat news embedding and graph embedding
        if self.concat:
            news = torch.stack([
                data.x[(data.batch == idx).nonzero().squeeze()[0]]
                for idx in range(data.num_graphs)
            ])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        x = F.log_softmax(self.fc2(x), dim=-1)
        return x

    def calculate_loss(self, data) -> torch.Tensor:
        output = self.forward(data)
        loss = F.nll_loss(output, data.y)
        return loss

    def predict(self, data_without_label) -> torch.Tensor:
        output = self.forward(data_without_label)
        return F.softmax(output, dim=1)
