import torch
import copy
from torch import nn, Tensor
import math
from typing import Dict, Union, Any
import torch.distributions as dist
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch

from faknow.model.model import AbstractModel


def ib_loss(g_enc_pos: Tensor, g_enc_neg: Tensor, g_enc: Tensor):
    """
    Compute contrastive loss
    Args:
        g_enc_pos(Tensor): positive augment data.
        g_enc_neg(Tensor): negative augment data.
        g_enc(Tensor): average representation. g_enc = (g_enc_pos + g_enc_neg) / 2.

    Returns:
        IB_Loss
    """
    margin = 2.0

    distance = torch.sqrt(torch.sum((g_enc_pos - g_enc_neg) ** 2))
    loss_cl = (1 - 1) * 0.5 * distance ** 2 + 1 * 0.5 * max(0, margin - distance) ** 2
    mean_H_IB = torch.zeros(g_enc.shape[1]).cuda()
    var_H_IB = torch.ones(g_enc.shape[1]).cuda()
    p_H_IB = dist.MultivariateNormal(mean_H_IB, torch.diag(var_H_IB))

    # Compute KL divergence
    mean_H_IB_given_X = g_enc.mean(dim=0).cuda()
    var_H_IB_given_X = F.softplus(g_enc.var(dim=0)).cuda()
    p_H_IB_given_X = dist.MultivariateNormal(mean_H_IB_given_X, torch.diag(var_H_IB_given_X))

    KL_loss = dist.kl_divergence(p_H_IB_given_X, p_H_IB).div(math.log(2)) / 128

    return loss_cl + 0.2 * KL_loss


class _GNNEncoder(nn.Module):
    """
    encoding the TF-IDF to interpretable embedding feature.
    """

    def __init__(self, num_features: int, embedding_num: int, num_gcn_layers: int):
        """
        num_feature(int): the num size of raw features.
        embedding_num(int): the num size of embedding features.
        num_gcn_layers(int): gcn layers num.
        """
        super(_GNNEncoder, self).__init__()

        self.num_gcn_layers = num_gcn_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_gcn_layers):
            if i:
                layer = nn.Sequential(nn.Linear(embedding_num, embedding_num), nn.ReLU(),
                                      nn.Linear(embedding_num, embedding_num))
            else:
                layer = nn.Sequential(nn.Linear(num_features, embedding_num), nn.ReLU(),
                                      nn.Linear(embedding_num, embedding_num))
            conv = GINConv(layer)
            bn = torch.nn.BatchNorm1d(embedding_num)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):

        x_one = copy.deepcopy(x)
        xs_one = []
        for i in range(self.num_gcn_layers):
            x_one = F.relu(self.convs[i](x_one, edge_index))
            xs_one.append(x_one)

        xpool_one = [global_mean_pool(x_one, batch) for x_one in xs_one]
        x_one = torch.cat(xpool_one, 1)
        return x_one, torch.cat(xs_one, 1)

    def get_embeddings(self, data: Data):

        with torch.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_embed, node_embed = self.forward(x, edge_index, batch)
        return node_embed


def generate_mask_node(x: Tensor, rate=0.6):
    """
    Generate the binary mask tensor with a certain probability.
    x(Tensor): node data.
    rate(float): drop rate. default=0.6
    """
    d = x.shape[1]
    mask = torch.zeros(d).bernoulli_(rate)
    sample_indices = torch.randint(x.shape[0], size=(1,))
    x_r = x[sample_indices]

    x_ib = (x_r - (x - x_r) * mask)
    return x_ib


class _FF(nn.Module):
    """
    Feed Forward Layer
    """

    def __init__(self, input_dim: int):
        """
        intput_dim(int): the features size of input.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x: Tensor):
        return self.block(x) + self.linear_shortcut(x)


class _BayesianLinear(nn.Module):
    """
    Bayesian Layer
    """

    def __init__(self, in_features: int, out_features: int):
        """
        in_features(int): the feature size of input.
        out_features(int): the feature size of output.
        """
        super(_BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weights_log_var = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mean = torch.nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = torch.nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights_mean, nonlinearity='relu')
        nn.init.constant_(self.weights_log_var, -10)
        nn.init.constant_(self.bias_mean, 0)
        nn.init.constant_(self.bias_log_var, -10)

    def forward(self, x: Tensor):
        weights = dist.Normal(self.weights_mean, self.weights_log_var.exp().sqrt()).rsample()
        bias = dist.Normal(self.bias_mean, self.bias_log_var.exp().sqrt()).rsample()
        return F.linear(x, weights, bias)


class Net(AbstractModel):
    """
    Pre-processed Net
    """

    def __init__(self, hidden_dim=64, num_gcn_layers=3):
        """
        hidden_dim(int): the feature size of hidden embedding. defult=64.
        num_gcn_layers(int): the gcn encoder layer num. default=3.
        """
        super(Net, self).__init__()

        self.embedding_dim = hidden_dim * num_gcn_layers
        self.encoder = _GNNEncoder(5000, hidden_dim, num_gcn_layers)

        self.local_d = _FF(self.embedding_dim)
        self.global_d = _FF(self.embedding_dim)
        self.mask_rate = nn.Parameter(torch.zeros(1))
        self.edge_mask = nn.Parameter(torch.zeros(1))
        self.nn = nn.Sequential(
            nn.Linear(in_features=5000, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def generate_drop_edge(self, x: Tensor, edgeindex: Tensor):
        """
        Generate dropped edge to augment data.

        """
        Z = self.nn(x)
        pi = torch.sigmoid(torch.matmul(Z, torch.t(Z)))
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        edgeindex_ib = []
        for i in range(len(row)):
            u, v = row[i], col[i]
            if torch.distributions.Bernoulli(pi[u, v]).sample() == 1:
                edgeindex_ib.append(i)
        row_ib = [row[i] for i in edgeindex_ib]
        col_ib = [col[i] for i in edgeindex_ib]
        drop_edgeindex = [row_ib, col_ib]
        return torch.LongTensor(drop_edgeindex)

    def forward(self, data: Data):

        x_pos_one = generate_mask_node(data.x)
        x_pos_two = generate_mask_node(data.x)
        dropped_edge_one = self.generate_drop_edge(data.x, data.edge_index)
        dropped_edge_two = self.generate_drop_edge(data.x, data.edge_index)

        _, M = self.encoder(data.x, data.edge_index, data.batch)

        y_pos_one, _ = self.encoder(x_pos_one, dropped_edge_one, data.batch)
        y_pos_two, _ = self.encoder(x_pos_two, dropped_edge_two, data.batch)
        # the encoded node feature matrix, corresponds to X in the L_ssl equation
        # Compute representation for each augmented graph
        g_enc_pos = self.global_d(y_pos_one)
        g_enc_neg = self.global_d(y_pos_two)
        # compute average representation for graph pairs
        g_enc = (g_enc_pos + g_enc_neg) / 2
        IB_loss = ib_loss(g_enc_pos, g_enc_neg, g_enc)

        return IB_loss

    def calculate_loss(self, data: Batch) -> Tensor:
        """
        calculate IB_loss

        Args:
            data(Batch): batch data

        Returns:
            torch.Tensor: IB_Loss
        """
        return self.forward(data)


class TRUSTRD(AbstractModel):
    r"""
    Towards Trustworthy Rumor Detection with Interpretable Graph Structural Learning, CIKM 2023
    paper: https://dl.acm.org/doi/10.1145/3583780.3615228
    code: https://github.com/Anonymous4ScienceAuthor/TrustRD
    """

    def __init__(self, encoder: AbstractModel, in_feature=192, hid_feature=64, num_classes=4,
                 sigma_m=0.1, eta=0.4, zeta=0.02):
        """
        encoder(AbstractModel): the model of trained Net.
        in_feature(int): the feature size of input. default=192.
        hid_feature(int): the feature size of hidden embedding. default=64.
        num_classes(int): the num of class. default=4.
        sigma_m(float): data perturbation Standard Deviation. default=0.1
        eta(float): data perturbation weight. default=0.4.
        zeta(float): parameter perturbation weight. default=0.02
        """
        super(TRUSTRD, self).__init__()
        self.encoder = encoder
        self.sigma_m = sigma_m
        self.eta = eta
        self.zeta = zeta

        self.linear_one = _BayesianLinear(5000 * 2, 2 * hid_feature)
        self.linear_two = _BayesianLinear(2 * hid_feature, hid_feature)
        self.linear_three = _BayesianLinear(in_feature, hid_feature)

        self.linear_transform = _BayesianLinear(hid_feature * 2, 4)
        self.prelu = nn.PReLU()
        self.num_classes = num_classes
        self.uncertainty_weight = 0.2
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embed: Tensor, data: Data):
        """
        Args:
            embed(Tensor): embedding features processed by Net.
            data(Data): batch data.

        Returns:
              Tensor: [predict_result, last_prob, kl_div]
        """
        ori = scatter_mean(data.x, data.batch, dim=0)
        root = data.x[data.rootindex]
        ori = torch.cat((ori, root), dim=1)
        ori = self.linear_one(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)
        ori = self.linear_two(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)

        x = scatter_mean(embed, data.batch, dim=0)
        x = self.linear_three(x)
        x = F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        out = torch.cat((x, ori), dim=1)
        out = self.linear_transform(out)

        pred_probs = []
        for i in range(10):
            x = F.log_softmax(out, dim=1)
            pred_prob = F.softmax(x, dim=1)
            pred_probs.append(pred_prob)
        mean_pred_prob = torch.stack(pred_probs).mean(dim=0)
        x = torch.log(mean_pred_prob)
        return x

    def calculate_loss(self, data: Batch) -> Dict[str, Union[float, Any]]:
        """
        calculate loss

        Args:
            data(Batch): batch data

        Returns:
            dict: loss dict with key 'total_loss', 'pred_loss', 'kl_loss', 'para_loss', 'data_loss'
        """
        kl_div = torch.zeros(1)
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl_div += module.kl_loss()

        self.encoder.eval()
        _, batch_embed = self.encoder.encoder(data.x, data.edge_index, data.batch)
        noise = torch.randn_like(batch_embed) * self.sigma_m
        noisy_embed = batch_embed + self.eta * noise
        loss_data = F.mse_loss(self.forward(noisy_embed, data),
                               self.forward(batch_embed, data))
        model_copy = copy.deepcopy(self)

        with torch.no_grad():
            for param, param_copy in zip(self.parameters(), model_copy.parameters()):
                noise = torch.randn_like(param)
                noise = self.zeta * noise / noise.norm(p=2)
                param_copy.data.add_(noise)
        loss_para = F.mse_loss(self.forward(batch_embed, data),
                               model_copy(batch_embed, data))
        out_labels = self.forward(batch_embed, data)
        finalloss = F.nll_loss(out_labels, data.y)

        loss = finalloss + 0.5 * kl_div + 0.2 * loss_para + 0.2 * loss_data

        return {'total_loss': loss, 'pred_loss': finalloss, 'kl_loss': kl_div,
                'para_loss': loss_para, 'data_loss': loss_data}

    def predict(self, data_without_label: Batch) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label(Batch): batch data

        Returns:
            Tensor: predication probability, shape=(num_graphs, 2)
        """
        self.encoder.eval()
        batch_embed = self.encoder.encoder.get_embeddings(data_without_label)
        pred_out = self.forward(batch_embed, data_without_label)

        return pred_out
