from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch import nn
from faknow.model.model import AbstractModel
from faknow.data.dataset.fang_dataset import FangDataset
from scipy import special
from collections import Counter
from torch.utils.data import Dataset

import torch.nn.functional as F
import numpy as np
import torch
import random


class _GraphSAGE(nn.Module):
    """
    graphsage model with 2 conv layers
    """

    def __init__(self, input_size: int, output_size: int, device='cpu'):
        super(_GraphSAGE, self).__init__()

        self.conv = nn.ModuleList()
        self.conv.append(SAGEConv(input_size, output_size))
        self.conv.append(SAGEConv(output_size, output_size))

        self.device = device
        self.num_layers = 2

    def forward(self, nodes: torch.Tensor, edge_list: torch.Tensor):

        for i, (edge_index, _, size) in enumerate(edge_list):
            nodes_target = nodes[:size[1]]
            nodes = self.conv[i]((nodes, nodes_target), edge_index)
            nodes = torch.tanh(nodes)

        return nodes


class _StanceClassifier(nn.Module):
    """
    Stance Classifier for users and news features
    """

    def __init__(self, embedding_size: int, num_stance: int, num_hidden: int):
        """
        Args:
            embedding_size(int): input dim
            num_stance(int): stance num
            num_hidden(int): out dim of every stance
        """
        super(_StanceClassifier, self).__init__()
        self.num_stance = num_stance
        self.num_hidden = num_hidden
        self.user_proj = nn.Linear(embedding_size, num_stance * num_hidden)
        self.news_proj = nn.Linear(embedding_size, num_stance * num_hidden)
        self.norm_coeff = np.sqrt(num_hidden)

    def forward(self, user: torch.Tensor, news: torch.Tensor):
        """
        Args:
            user(torch.Tensor): user data
            news(torch.Tensor): news data
        """
        h_user = self.user_proj(user)
        h_news = self.news_proj(news)
        e = (h_user * h_news).view(-1, self.num_stance, self.num_hidden)
        stance_pred = e.sum(-1) / self.norm_coeff
        return stance_pred


class _FakeNewsClassifier(nn.Module):
    """
    Fakenews classifier for news features after inferred by graphsage
    """

    def __init__(self, embedding_size: int, hidden_size: int, num_stance: int, timestamp_size: int, num_classes: int,
                 dropout: float):
        """
        Args:
            embedding_size(int): features size of input.
            hidden_size(int): the hidden_size of lstm.  fang's default=embedding_size/2
            num_stance(int): the total stance users takes for news.
            timestamp_size(int): the timestamp's feature size.
            num_classes(int): the labels num.
        """

        super(_FakeNewsClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=embedding_size + num_stance + timestamp_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(1 - dropout)

        # attention mechanism
        self.aligned_attn_proj = nn.Linear(in_features=hidden_size * 2,
                                           out_features=embedding_size,
                                           bias=False)
        self.meta_attn_proj = nn.Linear(in_features=num_stance + timestamp_size,
                                        out_features=1,
                                        bias=False)

        last_input_dim = 2 * hidden_size + embedding_size
        self.output_layer = nn.Linear(last_input_dim, num_classes)

    def forward(self, news: torch.Tensor, source: torch.Tensor, user: torch.Tensor, stances: torch.Tensor,
                timestamps: torch.Tensor, masked_attn: torch.Tensor):
        """
        Args:
            news(torch.Tensor):  news features.
            source(torch.Tensor): source features.
            user(torch.Tensor): users features.
            stances(torch.Tensor): stance features.
            timestamps(torch.Tensor): timestamp features.
            masked_attn(torch.Tensor): masked attention of entity.
        """
        engage_inputs = torch.cat([user, stances, timestamps], dim=-1)
        hidden_states, _ = self.lstm(engage_inputs)

        # attention
        projected_hidden_states = self.aligned_attn_proj(hidden_states)
        aligned_attentions = torch.bmm(projected_hidden_states, news.unsqueeze(-1)).squeeze(-1)
        meta_inputs = torch.cat([stances, timestamps], dim=-1).float()
        meta_attentions = self.meta_attn_proj(meta_inputs).squeeze(-1)
        aligned_attentions = torch.softmax(aligned_attentions + masked_attn, dim=-1)
        meta_attentions = torch.softmax(meta_attentions + masked_attn, dim=-1)
        attentions = (aligned_attentions + meta_attentions) / 2
        attn_embedding = torch.bmm(attentions.unsqueeze(1), hidden_states).squeeze(1)

        # MLP
        news_embed = torch.add(news, attn_embedding)
        news_source = torch.cat([source, news_embed], dim=1)

        return self.output_layer(news_source), attentions, news_embed


class FANG(AbstractModel):
    r"""
    FANG: Leveraging Social Context for Fake News Detection Using Graph Representation, CIKM 2020
    paper: https://dl.acm.org/doi/10.1145/3340531.3412046
    code: https://github.com/nguyenvanhoang7398/FANG
    """

    def __init__(self, fang_data: FangDataset,
                 input_size=100,
                 embedding_size=16,
                 num_stance=4,
                 num_stance_hidden=4,
                 timestamp_size=2,
                 num_classes=2,
                 dropout=0.1,
                 device='cpu'):
        """
        Args:
            fang_data(FangDataset): global graph information.
            input_size(int): raw features' embedding size. default=100.
            embedding_size(int): embedding size of fang and the output size of graphsage. default=16.
            num_stance(int):the total num of stance. default=4.
            num_stance_hidden(int):the feature size of stance(usually num_stance_hidden * num_stance = embedding_size). default=4.
            timestamp_size(int):the feature size of timestamp. default=2.
            num_classer(int): label num. default=2.
            dropout(float): dropout rate. default=0.1.
        """
        super(FANG, self).__init__()

        self.Q = 10  # used for compute unsupvised loss.
        self.fang_data = fang_data
        self.embedding_size = embedding_size
        self.device = device
        edge_list = [[], []]
        for edge in self.fang_data.edge_list:
            edge_list[0].append(edge[0])
            edge_list[1].append(edge[1])
        self.edge_list = torch.LongTensor(edge_list)

        self.graph_sage = _GraphSAGE(input_size, embedding_size, device)
        self.stance_classifier = _StanceClassifier(embedding_size, num_stance, num_stance_hidden)
        self.news_classifier = _FakeNewsClassifier(embedding_size, int(embedding_size / 2), num_stance,
                                                   timestamp_size, num_classes,
                                                   dropout)  # hidden_size = embedding_size/2, so that lstm output can add with graphsage output directly.

        self.news_loss = nn.CrossEntropyLoss()
        self.stance_loss = nn.CrossEntropyLoss()

    def preprocess_news_classification_data(self,
                                            nodes_batch: list,
                                            embed_size: int,
                                            n_stances: int,
                                            adj_lists: dict,
                                            stance_lists: dict,
                                            news_labels: dict,
                                            test=False):
        """
        Args:
            node_batch(list): the idx list of node to be processed.
            embed_size(int): the size of embedding features.
            n_stances(int): stance num.
            adj_lists(dict): adjacent information dict for every node.
            stance_list(dict): stance information dict for every node.
            news_labels(dict): news label dict.
            test(bool): if test or trai
        """
        smoothing_coeff = 1e-8  # to differentiate between 0 ts and padding value
        max_duration = 86400 * 7
        if not test:
            sample_size = [16, 4]
        else:
            sample_size = [-1, -1]
        sources, news = [], []
        all_engage_users, all_engage_stances, all_engage_ts, all_masked_attn, labels = [], [], [], [], []

        for node in nodes_batch:
            if node in self.fang_data.news_labels:
                news.append(node)
                _sources = [int(x.split("#")[0]) for x in adj_lists[node]]
                assert len(_sources) == 1, "only 1 source can publish the article"
                sources.append(_sources[0])
                engage_users, engage_stances, engage_ts, masked_attn = [], [], [], []
                for user, (stance, ts_mean, ts_std) in stance_lists[node]:
                    ts_mean = min(ts_mean, max_duration)
                    scaled_ts_mean = min(ts_mean, max_duration) / float(max_duration)
                    assert scaled_ts_mean >= 0
                    engage_users.append(user)
                    engage_stances.append(stance)
                    engage_ts.append([scaled_ts_mean + smoothing_coeff, ts_std + smoothing_coeff])
                    masked_attn.append(0)
                engage_users, engage_stances, engage_ts, masked_attn = \
                    engage_users[:100], engage_stances[:100], engage_ts[:100], \
                        masked_attn[:100]
                while len(engage_stances) < 100:
                    engage_stances.append(list(np.zeros(n_stances)))
                    engage_ts.append([0, 0])
                    masked_attn.append(-1000)
                all_engage_users.append(engage_users)
                all_engage_stances.append(engage_stances)
                all_engage_ts.append(engage_ts)
                all_masked_attn.append(masked_attn)
                labels.append(news_labels[node])

        news_labels = torch.LongTensor(labels).to(self.device)
        all_engage_stances = torch.FloatTensor(all_engage_stances).to(self.device)
        all_masked_attn = torch.FloatTensor(all_masked_attn).to(self.device)
        all_engage_ts = torch.FloatTensor(all_engage_ts).to(self.device)

        news_emb_batch = self.infer_embedding(news, sample_size).to(self.device)
        sources_emb_batch = self.infer_embedding(sources, sample_size).to(self.device)

        engage_user_emb_batch = [self.infer_embedding(e_users, sample_size) if len(e_users) > 0
                                 else torch.zeros(1, embed_size).to(self.device)
                                 for e_users in all_engage_users]
        for i, user_list in enumerate(engage_user_emb_batch):
            user_list = user_list[:100]
            if len(user_list) < 100:
                padding_mtx = torch.zeros((100 - len(user_list), embed_size)).to(self.device)
                engage_user_emb_batch[i] = torch.cat([user_list, padding_mtx], dim=0)
        engage_user_emb_batch = torch.stack(engage_user_emb_batch, dim=0)

        return sources_emb_batch, news_emb_batch, engage_user_emb_batch,  all_engage_stances,\
              all_engage_ts, news_labels, all_masked_attn, labels, all_engage_users

    def infer_embedding(self, node, sample_size):

        infer_loader = NeighborSampler(self.edge_list, node_idx=torch.tensor(node), sizes=sample_size,
                                       batch_size=len(node),shuffle=False)
        emb_batch = None
        for batch_size, n_ids,adjs in infer_loader:
            adjs = [adj.to(self.device) for adj in adjs]
            emb_batch = self.graph_sage(self.fang_data.feature_data[n_ids].to(self.device), adjs)

        return emb_batch

    def fetch_news_user_stance(self, news_nodes: list, stance_lists: dict):
        """
        fetch data required for stance prediction.

        Args:
            news_nodes(list): list of news node.
            stance_lists(dict): stance information dict for every node.
        """
        all_engaged_users, all_stance_labels, all_n_users = [], [], []
        for news in news_nodes:
            stance_info = stance_lists[news]
            for neigh, (stance, _, _) in stance_info:
                stance_idx = np.argmax(stance)
                all_engaged_users.append(neigh)
                all_stance_labels.append(stance_idx)
            all_n_users.append(len(stance_info))

        return all_engaged_users, all_stance_labels, all_n_users

    def extract_most_attended_users(self, news_attn: torch.Tensor, engage_users: list):
        """
        using attention to search the most attended user to compute unsupvised loss.

        Args:
            news_attn(torch.Tensor): users' attention for news.
            engage_users(list): all engaged users.
        """

        if news_attn is not None:
            top_attn_users = set()
            news_attn = news_attn.detach().cpu().numpy()
            attn_sorted_args = np.argsort(news_attn)
            top_args = attn_sorted_args[:, -3:]
            for i, news_engaged_users in enumerate(engage_users):
                attn_args = top_args[i]
                for idx in attn_args:
                    if idx < len(news_engaged_users):
                        top_attn_users.add(news_engaged_users[idx])
        else:
            all_engaged_users = []
            for i, news_engaged_users in enumerate(engage_users):
                all_engaged_users.extend(news_engaged_users)
            user_cnt = Counter(all_engaged_users)
            most_common_users = user_cnt.most_common()
            top_attn_users = set([u[0] for u in most_common_users])

        return top_attn_users

    def get_neigh_weights(self, node: int, node_only=False):
        """
        get node's neighbor.

        Args:
            node(int): node idx.
            node_only(bool): whether to get only node or both node and its edge information. default=False.
        """
        neighs = self.fang_data.adj_lists[int(node)]
        if node_only:
            return [int(x.split("#")[0]) for x in neighs]
        neigh_nodes, neigh_weights = [], []
        for x in neighs:
            x = x.split("#")
            neigh_nodes.append(int(x[0]))
            neigh_weights.append(float(x[1]))
        neigh_weights = special.softmax(neigh_weights)
        return neigh_nodes, neigh_weights

    def get_proximity_samples(self, nodes: list, num_pos: int, num_neg: int, neg_walk_len: int):
        """
        positive samping and negative sampling.

        Args:
            nodes(list): node idx list.
            num_pos(int): positive sampling num.
            num_neg(int): negative sampling num.
            neg_walk_len(int): walk step.
        """

        source_news = self.fang_data.news | self.fang_data.sources

        positive_pairs = []
        negative_pairs = []
        node_positive_pairs = {}
        node_negative_pairs = {}

        for node in nodes:
            node = int(node)
            homo_nodes = self.fang_data.users if node in self.fang_data.users else source_news
            neighbors, frontier = {node}, {node}
            for _ in range(neg_walk_len):
                current = set()
                for outer in frontier:
                    current |= set(self.get_neigh_weights(outer, node_only=True))
                frontier = current - neighbors
                neighbors |= current
            far_nodes = homo_nodes - neighbors
            neighbors -= {node}

            # update positive samples
            pos_samples = random.sample(neighbors, num_pos) if num_pos < len(neighbors) else neighbors
            pos_pairs = [(node, pos_node) for pos_node in pos_samples]
            positive_pairs.extend(pos_pairs)
            node_positive_pairs[node] = pos_pairs

            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            neg_pairs = [(node, neg_node) for neg_node in neg_samples]
            negative_pairs.extend(neg_pairs)
            node_negative_pairs[node] = neg_pairs

        return positive_pairs, node_positive_pairs, negative_pairs, node_negative_pairs

    def forward(self, data_batch: list, test=False):
        """
        Args:
            data_batch(list): node's idx list.
            test(bool): test or train. default=False
        """

        _news_sources_emb_batch, _news_emb_batch, _news_user_emb_batch, _news_stances_tensor, \
        _news_ts_tensor, _news_labels_tensor, _masked_attn_batch, _news_labels_batch, _engage_users_batch  \
            = self.preprocess_news_classification_data(data_batch,
                                                       self.embedding_size,
                                                       self.fang_data.n_stances,
                                                       self.fang_data.adj_lists,
                                                       self.fang_data.stance_lists,
                                                       self.fang_data.news_labels,
                                                       test=test)
        _news_logit_batch, _news_attention, _news_emb_batch = self.news_classifier(
            _news_emb_batch, _news_sources_emb_batch, _news_user_emb_batch, _news_stances_tensor,
            _news_ts_tensor, _masked_attn_batch)

        return _news_logit_batch, _news_attention, _news_emb_batch, _news_labels_tensor, _engage_users_batch


    def calculate_loss(self, data: torch.Tensor) -> dict:
        """
        calculate total loss

        Args:
            data(torch.Tensor): nodes' idx list

        Returns:
            Torch.Tensor: news_loss + stance_loss + unsup_loss
        """
        data = data.tolist()
        source_batch = [n for n in data if n in self.fang_data.sources]
        source_emb_batch = self.infer_embedding(source_batch, [16, 4]) \
            if len(source_batch) > 0 else None

        news_batch = [n for n in data if n in self.fang_data.news]
        train_news_batch = [n for n in news_batch if n in self.fang_data.train_idxs]
        val_test_news_batch = [n for n in news_batch if n not in self.fang_data.train_idxs]

        news_logit_batch, train_news_attn, train_news_emb_batch, news_labels_tensor, train_engage_users = \
            self.forward(train_news_batch)

        all_top_attn_users = set()
        train_top_attn_users = self.extract_most_attended_users(train_news_attn, train_engage_users)
        all_top_attn_users |= train_top_attn_users

        # news loss
        news_loss = self.news_loss(news_logit_batch, news_labels_tensor).mean()

        if len(val_test_news_batch) > 0:
            _, val_test_news_attn, val_test_news_emb_batch, _, val_test_engage_users = \
                self.forward(val_test_news_batch)

            val_test_top_attn_users = self.extract_most_attended_users(val_test_news_attn, val_test_engage_users)
            all_top_attn_users |= val_test_top_attn_users

        # stance loss
        engaged_news_batch = train_news_batch + val_test_news_batch
        engaged_users_batch, stance_labels_batch, all_n_users = self.fetch_news_user_stance(engaged_news_batch,
                                                                                self.fang_data.stance_lists)
        if len(engaged_news_batch) > 0:
            if len(train_news_batch) > 0 and len(val_test_news_batch) > 0:
                _engaged_news_emb_batch = torch.cat([train_news_emb_batch, val_test_news_emb_batch], dim=0)
            elif len(train_news_batch) > 0:
                _engaged_news_emb_batch = train_news_emb_batch
            elif len(val_test_news_batch) > 0:
                _engaged_news_emb_batch = val_test_news_emb_batch
            else:
                raise ValueError("Engaged news batch should not be empty")

        repeats = torch.LongTensor(all_n_users).to(self.device)
        engaged_news_embed_batch = torch.repeat_interleave(_engaged_news_emb_batch, repeats, dim=0)
        engaged_user_embed_batch = self.infer_embedding(engaged_users_batch, [16, 4])
        stance_logit_batch = self.stance_classifier(engaged_news_embed_batch, engaged_user_embed_batch)
        stance_labels_tensor = torch.LongTensor(stance_labels_batch).to(self.device)
        stance_loss = self.stance_loss(stance_logit_batch, stance_labels_tensor).mean()

        # unsupervised loss
        positive_pairs, node_positive_pairs, \
            negative_pairs, node_negative_pairs = self.get_proximity_samples(data,
                                                                             4,
                                                                             4,
                                                                             1)

        extended_news_source_batch = list(set([i for x in positive_pairs for i in x])
                                          | set([i for x in negative_pairs for i in x]))
        extended_news_source_batch = np.asarray(list(extended_news_source_batch))

        extended_source_batch = [n for n in extended_news_source_batch if n in self.fang_data.sources]
        extended_source_embedding_batch = self.infer_embedding(extended_source_batch, [16, 4]) \
                if len(extended_source_batch) > 0 else None
        extended_news_batch = [n for n in extended_news_source_batch
                               if n in self.fang_data.news_labels and n not in news_batch]
        extended_news_emb_batch = None

        if len(extended_news_batch) > 0:
            _, extended_news_attn, extended_news_emb_batch, _, extended_engage_users = \
                self.forward(extended_news_batch)

            extended_top_attn_users = self.extract_most_attended_users(extended_news_attn, extended_engage_users)
            all_top_attn_users |= extended_top_attn_users

        user_batch = list(all_top_attn_users)
        user_pos_pairs, user_node_pos_pairs, \
            user_neg_pairs, user_node_neg_pairs = self.get_proximity_samples(user_batch,
                                                                             4,
                                                                             4,
                                                                             1)
        positive_pairs.extend(user_pos_pairs)
        negative_pairs.extend(user_neg_pairs)
        node_positive_pairs.update(user_node_pos_pairs)
        node_negative_pairs.update(user_node_neg_pairs)

        user_batch = list(set([i for x in positive_pairs for i in x])
                          | set([i for x in negative_pairs for i in x]))
        user_batch = [n for n in user_batch if n in self.fang_data.users]
        user_emb_batch = self.infer_embedding(user_batch, [16, 4])\
                if len(user_batch) > 0 else None
        node_embedding_batch_list, nodes_batch = [], []
        if user_emb_batch is not None:
            node_embedding_batch_list.append(user_emb_batch)
            nodes_batch.extend(user_batch)
        if source_emb_batch is not None:
            node_embedding_batch_list.append(source_emb_batch)
            nodes_batch.extend(source_batch)
        if extended_source_embedding_batch is not None:
            node_embedding_batch_list.append(extended_source_embedding_batch)
            nodes_batch.extend(extended_source_batch)
        if train_news_emb_batch is not None:
            node_embedding_batch_list.append(train_news_emb_batch)
            nodes_batch.extend(train_news_batch)
        if val_test_news_emb_batch is not None:
            node_embedding_batch_list.append(val_test_news_emb_batch)
            nodes_batch.extend(val_test_news_batch)
        if extended_news_emb_batch is not None:
            node_embedding_batch_list.append(extended_news_emb_batch)
            nodes_batch.extend(extended_news_batch)
        node_embedding_batch = torch.cat(node_embedding_batch_list, dim=0)
        node2index = {n: i for i, n in enumerate(nodes_batch)}

        node_scores = []

        for node in data:
            pps = node_positive_pairs[node]
            nps = node_negative_pairs[node]

            neg_indexs = [list(x) for x in zip(*nps)]

            neg_node_indexs = [node2index[int(x)] for x in neg_indexs[0]]
            neg_neighbor_indexs = [node2index[int(x)] for x in neg_indexs[1]]
            neg_node_embeddings, neg_neighbor_embeddings = node_embedding_batch[neg_node_indexs], \
                node_embedding_batch[neg_neighbor_indexs]
            neg_sim_score = F.cosine_similarity(neg_node_embeddings, neg_neighbor_embeddings)
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_sim_score)), 0)

            pos_indexs = [list(x) for x in zip(*pps)]

            pos_node_indexs = [node2index[int(x)] for x in pos_indexs[0]]
            pos_neighbor_indexs = [node2index[int(x)] for x in pos_indexs[1]]
            pos_node_embeddings, pos_neighbor_embeddings = node_embedding_batch[pos_node_indexs], \
                node_embedding_batch[pos_neighbor_indexs]
            pos_sim_score = F.cosine_similarity(pos_node_embeddings, pos_neighbor_embeddings)
            pos_score = torch.log(torch.sigmoid(pos_sim_score))

            node_score = torch.mean(- pos_score - neg_score).view(1, -1)
            node_scores.append(node_score)

        unsup_loss = torch.mean(torch.cat(node_scores, 0))

        total_loss = news_loss + stance_loss + unsup_loss

        return {'total_loss': total_loss, 'news_loss': news_loss,
                'stance_loss': stance_loss, 'unsup_loss':unsup_loss}

    def predict(self, data_without_label: dict) -> torch.Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label(dict): batch data

        Returns:
            torch.Tensor: predict probability, shape=(batch_size, 2)
        """
        data_without_label = data_without_label['data']
        nodes = [int(n) for n in data_without_label]
        news_logit_batch, _, _, _, _ = self.forward(nodes, test=True)

        return news_logit_batch
