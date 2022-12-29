from template.model.model import AbstractModel
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool


class _BaseGNN(AbstractModel):
    def __init__(self, feature_size: int, hidden_size: int, concat=False):
        super(_BaseGNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.concat = concat

        if self.concat:
            self.fc0 = torch.nn.Linear(self.feature_size, self.hidden_size)
            self.fc1 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = global_max_pool(x, batch)

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


class GCN(_BaseGNN):
    def __init__(self, feature_size: int, hidden_size: int, concat=False):
        super().__init__(feature_size, hidden_size, concat)
        self.conv = GCNConv(self.feature_size, self.hidden_size)


class SAGE(_BaseGNN):
    def __init__(self, feature_size: int, hidden_size: int, concat=False):
        super().__init__(feature_size, hidden_size, concat)
        self.conv = SAGEConv(self.feature_size, self.hidden_size)


class GAT(_BaseGNN):
    def __init__(self, feature_size: int, hidden_size: int, concat=False):
        super().__init__(feature_size, hidden_size, concat)
        self.conv = GATConv(self.feature_size, self.hidden_size)
