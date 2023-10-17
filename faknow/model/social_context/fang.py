from torch_geometric.nn import SAGEConv
from torch import nn
from faknow.model.model import AbstractModel
from faknow.data.dataset.fang_dataset import FangDataset
from scipy import special
from collections import Counter

import torch.nn.functional as F
import numpy as np
import torch
import random


class _GraphSAGE(nn.Module):
    """
    graphsage model with 2 conv layers
    """

    def __init__(self, input_size: int, output_size: int):
        super(_GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(input_size, output_size)
        self.conv2 = SAGEConv(output_size, output_size)

    def forward(self, nodes: torch.Tensor, edge_list: torch.Tensor):
        nodes = self.conv1(nodes, edge_list)
        nodes = self.conv2(nodes, edge_list)

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

        return self.output_layer(news_source), attentions


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
                 dropout=0.1):
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
        self.graph_sage = _GraphSAGE(input_size, embedding_size)
        self.stance_classifier = _StanceClassifier(embedding_size, num_stance, num_stance_hidden)
        self.news_classifier = _FakeNewsClassifier(embedding_size, int(embedding_size / 2), num_stance,
                                                   timestamp_size, num_classes,
                                                   dropout)  # hidden_size = embedding_size/2, so that lstm output can add with graphsage output directly.

        self.news_loss = nn.CrossEntropyLoss()
        self.stance_loss = nn.CrossEntropyLoss()

    def preprocess_news_classification_data(self,
                                            nodes_batch: list,
                                            n_stances: int,
                                            adj_lists: dict,
                                            stance_lists: dict,
                                            news_labels: dict):
        """
        Args:
            node_batch(list): the idx list of node to be processed.
            n_stances(int): stance num.
            adj_lists(dict): adjacent information dict for every node.
            stance_list(dict): stance information dict for every node.
            news_labels(dict): news label dict.
        """
        smoothing_coeff = 1e-8  # to differentiate between 0 ts and padding value
        max_duration = 86400 * 7
        sources, news = [], []
        all_engage_users, all_engage_stances, all_engage_ts, all_masked_attn, labels = [], [], [], [], []

        for node in nodes_batch:
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

        news_labels = torch.LongTensor(labels)
        all_engage_stances = torch.LongTensor(all_engage_stances)
        all_masked_attn = torch.LongTensor(all_masked_attn)
        all_engage_ts = torch.LongTensor(all_engage_ts)

        return sources, news, all_engage_users, all_engage_stances, \
            all_engage_ts, news_labels, all_masked_attn

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

    def forward(self, data_batch: list):
        """
        Args:
            data_batch(list): node's idx list.
        """

        news_batch = [n for n in data_batch if n in self.fang_data.news]

        edge_list = [[], []]
        for edge in self.fang_data.edge_list:
            edge_list[0].append(edge[0])
            edge_list[1].append(edge[1])
        edge_list = torch.LongTensor(edge_list)

        entity_embedding = self.graph_sage(self.fang_data.feature_data, edge_list)

        sources, news, all_engage_users, all_engage_stances, all_engage_ts, news_label, \
            all_masked_atten = self.preprocess_news_classification_data(news_batch,
                                                                        self.fang_data.n_stances,
                                                                        self.fang_data.adj_lists,
                                                                        self.fang_data.stance_lists,
                                                                        self.fang_data.news_labels)
        news_embedding = entity_embedding[news, :]
        sources_embedding = entity_embedding[sources, :]
        engage_users_embedding = [entity_embedding[e_users, :] if len(e_users) > 0
                                  else torch.zeros(1, self.embedding_size)
                                  for e_users in all_engage_users]

        for i, user_list in enumerate(engage_users_embedding):
            user_list = user_list[:100]
            if len(user_list) < 100:
                padding_mtx = torch.zeros(100 - len(user_list), self.embedding_size)
                engage_users_embedding[i] = torch.cat([user_list, padding_mtx], dim=0)
        engage_users_embedding = torch.stack(engage_users_embedding, dim=0)

        logits, news_attention = self.news_classifier(news_embedding, sources_embedding,
                                                      engage_users_embedding,
                                                      all_engage_stances,
                                                      all_engage_ts,
                                                      all_masked_atten)

        return logits, entity_embedding, news_label, news_embedding, news, sources, all_engage_users, news_attention

    def calculate_loss(self, data: list) -> torch.Tensor:
        """
        calculate total loss

        Args:
            data(list): nodes' idx list

        Returns:
            Torch.Tensor: news_loss + stance_loss + unsup_loss
        """

        nodes = data

        nodes = [int(n) for n in nodes if int(n) in self.fang_data.news]

        logits, entity_embedding, news_label, \
            news_embedding, news, sources, all_engaged_user, news_attention = self.forward(nodes)

        all_top_attn_users = set()
        train_top_attn_users = self.extract_most_attended_users(news_attention, all_engaged_user)
        all_top_attn_users |= train_top_attn_users

        # news loss
        news_loss = self.news_loss(logits, news_label)

        # stance loss
        engaged_users, stance_labels, all_n_users = self.fetch_news_user_stance(nodes,
                                                                                self.fang_data.stance_lists)
        repeats = torch.LongTensor(all_n_users)
        engaged_news = torch.repeat_interleave(news_embedding, repeats, dim=0)
        engaged_users = entity_embedding[engaged_users, :]
        stance_logits = self.stance_classifier(engaged_news, engaged_users)
        stance_labels = torch.LongTensor(stance_labels)
        stance_loss = self.stance_loss(stance_logits, stance_labels)

        # unsupervised loss
        positive_pairs, node_positive_pairs, \
            negative_pairs, node_negative_pairs = self.get_proximity_samples(nodes,
                                                                             4,
                                                                             4,
                                                                             1)

        extended_news_source_batch = list(set([i for x in positive_pairs for i in x])
                                          | set([i for x in negative_pairs for i in x]))
        extended_news_source_batch = np.asarray(list(extended_news_source_batch))

        extended_source_batch = [n for n in extended_news_source_batch if n in self.fang_data.sources]
        extended_source_embedding_batch = entity_embedding[extended_source_batch, :]
        extended_news_batch = [n for n in extended_news_source_batch if n in self.fang_data.news and n not in nodes]
        extended_news_embedding_batch = entity_embedding[extended_news_batch]
        _, _, _, _, _, _, extended_all_engaged_users, extended_news_attention = self.forward(extended_news_batch)
        extended_top_users = self.extract_most_attended_users(extended_news_attention,
                                                              extended_all_engaged_users)
        all_top_attn_users |= extended_top_users

        user_batch = list(all_top_attn_users)
        user_pos_pairs, user_node_pos_pairs, \
            user_neg_pairs, user_node_neg_pairs = self.get_proximity_samples(user_batch,
                                                                             4,
                                                                             4,
                                                                             1)
        positive_pairs.extend(user_pos_pairs)
        negative_pairs.extend(user_neg_pairs)
        node_positive_pairs = node_positive_pairs | user_node_pos_pairs
        node_negative_pairs = node_negative_pairs | user_node_neg_pairs

        user_batch = list(set([i for x in positive_pairs for i in x])
                          | set([i for x in negative_pairs for i in x]))
        user_batch = [n for n in user_batch if n in self.fang_data.users]
        user_embedding = entity_embedding[user_batch, :]
        node_embedding_batch_list, nodes_batch = [], []
        if user_embedding is not None:
            node_embedding_batch_list.append(user_embedding)
            nodes_batch.extend(user_batch)
        if sources is not None:
            node_embedding_batch_list.append(entity_embedding[sources])
            nodes_batch.extend(sources)
        if extended_source_embedding_batch is not None:
            node_embedding_batch_list.append(extended_source_embedding_batch)
            nodes_batch.extend(extended_source_batch)
        if news_embedding is not None:
            node_embedding_batch_list.append(news_embedding)
            nodes_batch.extend(news)
        if extended_news_embedding_batch is not None:
            node_embedding_batch_list.append(extended_news_embedding_batch)
            nodes_batch.extend(extended_news_batch)
        node_embedding_batch = torch.cat(node_embedding_batch_list, dim=0)
        node2index = {n: i for i, n in enumerate(nodes_batch)}

        node_scores = []

        for node in data:
            pps = node_positive_pairs[int(node)]
            nps = node_negative_pairs[int(node)]

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

        return news_loss + stance_loss + unsup_loss

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
        logits, _, _, _, _, _, _, _ = self.forward(nodes)

        return logits
