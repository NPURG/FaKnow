import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GCNConv
from collections import OrderedDict
from torch_scatter import scatter_mean
from faknow.model.model import AbstractModel


class _RumorGCN(nn.Module):
    """
    GCN layer with edged weighted inferring
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, edge_num: int, dropout: float, device: str):
        """
        input_size(int): the feature size of input.
        hidden_size(int): the feature size of hidden embedding.
        output_size(int): the feature size of output embedding.
        edge_num(int): the num of edge type.
        dropout(float): dropout rate.
        device(str): device to compute.
        """
        super(_RumorGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.edge_num = edge_num
        self.dropout = dropout
        self.device = device

        self.conv1 = GCNConv(self.input_size, self.hidden_size)
        self.conv2 = GCNConv(self.input_size + self.hidden_size, self.output_size)
        self.sim_network = nn.Sequential(self.create_network('sim_val'))
        self.W_mean = nn.Sequential(self.create_network('W_mean'))
        self.W_bias = nn.Sequential(self.create_network('W_bias'))
        self.B_mean = nn.Sequential(self.create_network('B_mean'))
        self.B_bias = nn.Sequential(self.create_network('B_bias'))
        self.fc1 = nn.Linear(self.hidden_size, self.edge_num, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.edge_num, bias=False)
        self.dropout = nn.Dropout(self.dropout)
        self.unsup_loss = nn.KLDivLoss(reduction='batchmean')
        self.bn1 = nn.BatchNorm1d(self.hidden_size + self.input_size)

    def forward(self, node_features: Tensor, edge_index: Tensor, root_index: Tensor, batch_size: Tensor) -> tuple[
        Tensor, Tensor]:
        """
        Args:
            node_features(Tensor): features of graph nodes.
            edge_index(Tensor): adjacent matrix in COO format.
            root_index(Tensor): index of root in each claim(graph).
            batch_size(Tensor): vector mapping each node to its respective graph in the batch.

        Returns:
            tuple[Tensor, Tensor]: node features, edge loss.
        """
        node_features1 = copy.copy(node_features.float())
        node_features = self.conv1(node_features, edge_index)
        node_features2 = copy.copy(node_features)

        edge_loss, edge_pred = self.edge_infer(node_features, edge_index)

        root_extend = torch.zeros(len(batch_size), node_features1.size(1)).to(self.device)
        batch_num = max(batch_size) + 1
        for num_batch in range(batch_num):
            index = (torch.eq(batch_size, num_batch))
            root_extend[index] = node_features1[root_index[num_batch]]
        node_features = torch.cat((node_features, root_extend), 1)

        node_features = self.bn1(node_features)
        node_features = F.relu(node_features)
        node_features = self.conv2(node_features, edge_index, edge_pred)
        node_features = F.relu(node_features)
        root_extend = torch.zeros(len(batch_size), node_features2.size(1)).to(self.device)
        for num_batch in range(batch_num):
            index = (torch.eq(batch_size, num_batch))
            root_extend[index] = node_features2[root_index[num_batch]]
        node_features = torch.cat((node_features, root_extend), 1)

        node_features = scatter_mean(node_features, batch_size, 0)

        return node_features, edge_loss

    def create_network(self, name: str) -> OrderedDict:
        """
        create conv layer and activation layer

        Args:
            name(str): network name.

        Returns:
            OrderedDict: layer_list - dict of network
        """
        layer_list = OrderedDict()
        layer_list[name + 'conv'] = nn.Conv1d(in_channels=self.hidden_size,
                                              out_channels=self.hidden_size,
                                              kernel_size=1,
                                              bias=False)
        layer_list[name + 'norm'] = nn.BatchNorm1d(num_features=self.hidden_size)
        layer_list[name + 'relu'] = nn.LeakyReLU()
        layer_list[name + 'conv_out'] = nn.Conv1d(in_channels=self.hidden_size,
                                                  out_channels=1,
                                                  kernel_size=1)
        return layer_list

    def edge_infer(self, node_features: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """
        infer edge weight and unsupervised loss

        Args:
            node_features(Tensor):  features of graph nodes.
            edge_index(Tensor): adjacent matrix in COO format.

        Returns:
            tuple[Tensor, Tensor]: unsup_loss, edge_pred
        """
        row, col = edge_index[0], edge_index[1]
        x_i = node_features[row - 1].unsqueeze(2)
        x_j = node_features[col - 1].unsqueeze(1)
        x_ij = torch.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = torch.sigmoid(edge_pred)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = torch.log((sim_val ** 2) * torch.exp(w_bias) + torch.exp(b_bias))
        logit_var = torch.abs(logit_var)
        edge_y = torch.normal(logit_mean, logit_var)
        edge_y = torch.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)

        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        unsup_loss = self.unsup_loss(logp_x, p_y)
        return unsup_loss, torch.mean(edge_pred, dim=-1).squeeze(1)


class EBGCN(AbstractModel):
    r"""
    Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional Networks for Rumor Detection，ACL 2021
    paper: https://arxiv.org/pdf/2107.11934.pdf
    code: https://github.com/weilingwei96/EBGCN
    """

    def __init__(self,
                 input_size=5000,
                 hidden_size=64,
                 output_size=64,
                 edge_num=2,
                 dropout=0.5,
                 num_class=4,
                 edge_loss_weight=0.2,
                 device='cpu'):
        """
        Args:
            input_size(int): the feature size of input. default=5000.
            hidden_size(int): the feature size of hidden embedding. default=64.
            output_size(int): the feature size of output embedding. default=64.
            edge_num(int): the num of edge type. default=2.
            dropout(float): dropout rate. default=0.5.
            num_class(int): the num of output type. default=4
            edge_loss_weight(float): the weight of edge loss. default=0.2.
            device(str): device. default='cpu'.
        """
        super(EBGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.edge_num = edge_num
        self.dropout = dropout
        self.num_class = num_class
        self.edge_loss_weight = edge_loss_weight
        self.device = device

        self.TDRumorGCN = _RumorGCN(self.input_size, self.hidden_size, self.output_size,
                                    self.edge_num, self.dropout, self.device)
        self.BURumorGCN = _RumorGCN(self.input_size, self.hidden_size, self.output_size,
                                    self.edge_num, self.dropout, self.device)
        self.fc = nn.Linear((self.hidden_size + self.output_size) * 2, self.num_class)

    def forward(self, node_features: Tensor, TD_edge_index: Tensor,
                BU_edge_index: Tensor, root_index: Tensor,
                batch_size: Tensor):
        """
        Args:
            node_features(Tensor): feature of node in claims.
            TD_edge_index(Tensor): directed adjacent martix in COO format from top to bottom.
            BU_edge_index(Tensor): directed adjacent martix in COO format from bottom to top.
            root_index(Tensor): index of root news in claims.
            batch_size(Tensor): vector mapping each node to its respective graph in the batch.

        Returns:
            output(Tensor): predict output.
            TD_edge_loss(Tensor): edge loss of graph with direction from top to bottom.
            BU_edge_loss(Tensor): edge loss of graph with direction from bottom to top.
        """

        TD_node_features, TD_edge_loss = self.TDRumorGCN(node_features, TD_edge_index,
                                                         root_index, batch_size)
        BU_node_features, BU_edge_loss = self.BURumorGCN(node_features, BU_edge_index,
                                                         root_index, batch_size)

        self.fc_input = torch.cat((BU_node_features, TD_node_features), 1)
        output = self.fc(self.fc_input)
        output = F.log_softmax(output, dim=1)
        return output, TD_edge_loss, BU_edge_loss

    def calculate_loss(self, data: dict) -> Tensor:
        """
        calculate loss for EBGCN

        Args:
            data (dict): batch data dict

        Returns:
            Tensor: loss
        """
        node_features = data.x
        TD_edge_index = data.edge_index
        BU_edge_index = data.BU_edge_index
        root_index = data.root_index
        label = data.y
        batch_size = data.batch

        output, TD_edge_loss, BU_edge_loss = self.forward(node_features, TD_edge_index, BU_edge_index,
                                                          root_index, batch_size)
        pred_loss = F.nll_loss(output, label)
        total_loss = pred_loss + self.edge_loss_weight * (TD_edge_loss + BU_edge_loss)

        return total_loss

    def predict(self, data_without_label: dict) -> Tensor:
        """
         predict the probability of being fake news

        Args:
            data_without_label: batch data

        Returns:
            Tensor: softmax probability, shape=(batch_size, num_classes)
        """
        node_features = data_without_label.x
        TD_edge_index = data_without_label.edge_index
        BU_edge_index = data_without_label.BU_edge_index
        root_index = data_without_label.root_index
        data_batch = data_without_label.batch
        output, _, _ = self.forward(node_features, TD_edge_index, BU_edge_index,
                                    root_index, data_batch)

        return output
