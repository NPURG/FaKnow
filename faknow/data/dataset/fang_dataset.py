import numpy
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import torch
import os
import json
import csv
from torch.utils.data import Dataset

NAME_WEIGHT_DELIMITER = "#"
STANCE_DELIMITER = "_"
USER_TAG, NEWS_TAG, SOURCE_TAG, COMMUNITY_TAG = "user", "news", "source", "community"

def encode_class_idx_label(label_map: dict):
    """ encode label to one_hot embedding"""
    classes = list(sorted(set(label_map.values())))
    class2idx = {c: i for i, c in enumerate(classes)}
    labels_onehot_map = {k: class2idx[v] for k, v in label_map.items()}
    return labels_onehot_map, len(classes), class2idx

def row_normalize(mtx: numpy.array):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mtx.sum(axis=1))
    sum_inv = np.power(row_sum, -1).flatten()
    sum_inv[np.isinf(sum_inv)] = 0
    row_mtx_inv = sp.diags(sum_inv)
    return row_mtx_inv.dot(mtx)

def is_tag(entity_type: str, entity: str):
    """ find whether the entity is tagged."""
    return entity.startswith(entity_type)


def read_csv(path: str, load_header: bool=False, delimiter: str= ","):
    content = []
    with open(path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=delimiter, quotechar='"')
        if load_header:
            [content.append(row) for row in csv_reader]
        else:
            [content.append(row) for i, row in enumerate(csv_reader) if i > 0]
    return content

def load_text_as_list(input_path: str):
    with open(input_path, 'r', encoding="utf-8") as f:
        return f.read().splitlines()

def load_json(input_path: str):
    with open(input_path, "rb") as f:
        return json.load(f)

class FangDataset(Dataset):
    """
    construct global graph from tsv, txt and csv
    """
    def __init__(self, root_path: str):

        self.root_path = root_path

        self.adj_lists, self.edge_list, self.stance_lists = defaultdict(set), [], defaultdict(list)

        self.feature_data = None

        self.n_stances = None
        self.news_labels = None
        self.n_news_labels = None
        self.class2idx = {}
        self.entities, self.node_name_idx_map = None, None
        self.news_label_map = None
        self.train_idxs, self.dev_idxs, self.test_idxs = [], [], []
        self.news, self.users, self.sources = set(), set(), set()
        self.rep_entities = None

        self.load()

    def load_and_update_adj_lists(self, edge_file: str):
        """
        construct adjacent_list from edge information doc
        """

        edge_csv_content = read_csv(edge_file, True, delimiter="\t")

        for row in edge_csv_content:
            weight = 1. if len(row) == 2 else float(row[2])
            node_src, node_dest = self.node_name_idx_map[row[0]], self.node_name_idx_map[row[1]]
            self.adj_lists[node_src].add(str(node_dest) + NAME_WEIGHT_DELIMITER + str(weight))
            self.adj_lists[node_dest].add(str(node_src) + NAME_WEIGHT_DELIMITER + str(weight))
            self.edge_list.append((node_src, node_dest, weight))

    def load_stance_map(self, stance_file: str):
        """
        get stance feature from stance information doc
        """
        stance_content = read_csv(stance_file, True, delimiter="\t")
        stance_map = defaultdict(list)
        for row in stance_content:
            weight = 1. if len(row) == 3 else float(row[2])
            ts_mean = float(row[2]) if len(row) == 3 else float(row[3])
            ts_std = 0. if len(row) == 3 else float(row[4])
            user, news = self.node_name_idx_map[row[0]], self.node_name_idx_map[row[1]]
            stance_map[news].append([user, weight, ts_mean, ts_std])
        for news, engagements in stance_map.items():
            stance_map[news] = sorted(engagements, key=lambda x: x[2])

        return stance_map

    def get_news_label_map(self,news_info_path: str):
        news_info_data = read_csv(news_info_path, True, "\t")
        news_label_map = {row[0]: row[1] for row in news_info_data}
        return news_label_map

    def get_train_val_test_labels_nodes(self, entities: list, news_label_map: dict):
        idx_train, idx_val, idx_test = [], [], []
        train_test_path = os.path.join(self.root_path, "train_test.json")
        train_test_val = load_json(train_test_path)
        train_news, val_news, test_news = train_test_val["train"], train_test_val["val"], train_test_val["test"]
        for i, e in enumerate(entities):
            if e in news_label_map:
                if e in train_news:
                    idx_train.append(i)
                elif e in val_news:
                    idx_val.append(i)
                elif e in test_news:
                    idx_test.append(i)

        return idx_train, idx_val, idx_test

    def load(self):
        """
        load all doc from root path.
        """

        entity_path = os.path.join(self.root_path, "entities.txt")
        entity_feature_path = os.path.join(self.root_path, "entity_features.tsv")
        source_citation_path = os.path.join(self.root_path, "source_citation.tsv")
        source_publication_path = os.path.join(self.root_path, "source_publication.tsv")
        user_relationship_path = os.path.join(self.root_path, "user_relationships.tsv")
        news_info_path = os.path.join(self.root_path, "news_info.tsv")

        report_stance_path = os.path.join(self.root_path, "report.tsv")
        support_neutral_stance_path = os.path.join(self.root_path, "support_neutral.tsv")
        support_negative_stance_path = os.path.join(self.root_path, "support_negative.tsv")
        deny_stance_path = os.path.join(self.root_path, "deny.tsv")

        self.entities = load_text_as_list(entity_path)
        self.news_label_map = self.get_news_label_map(news_info_path)
        self.train_idxs, self.dev_idxs, self.test_idxs = \
            self.get_train_val_test_labels_nodes(self.entities, self.news_label_map)

        node_names = np.array(self.entities, dtype=np.dtype(str))
        self.node_name_idx_map = {j: i for i, j in enumerate(node_names)}
        for e in self.entities:
            e_idx = self.node_name_idx_map[e]
            if is_tag(NEWS_TAG, e):
                self.news.add(e_idx)
            if is_tag(USER_TAG, e):
                self.users.add(e_idx)
            if is_tag(SOURCE_TAG, e):
                self.sources.add(e_idx)

        self.news_label_map = {self.node_name_idx_map[k]: v for k, v in self.news_label_map.items()}
        self.news_labels, self.n_news_labels, self.class2idx = encode_class_idx_label(self.news_label_map)

        feature_content = read_csv(entity_feature_path, True, delimiter="\t")
        feature_data = [[float(x) for x in row[1:]] for row in feature_content]
        self.feature_data = row_normalize(np.asarray(feature_data))
        self.feature_data = torch.FloatTensor(self.feature_data)

        self.load_and_update_adj_lists(source_citation_path)
        self.load_and_update_adj_lists(source_publication_path)
        self.load_and_update_adj_lists(user_relationship_path)

        all_stance_map = []
        all_stance_map.append(self.load_stance_map(deny_stance_path))
        all_stance_map.append(self.load_stance_map(report_stance_path))
        all_stance_map.append(self.load_stance_map(support_neutral_stance_path))
        all_stance_map.append(self.load_stance_map(support_negative_stance_path))
        all_engaged_news = set()
        self.n_stances = len(all_stance_map)
        for stance_map in all_stance_map:
            all_engaged_news |= set(stance_map.keys())
        news_user_stance_map = {}
        for news in all_engaged_news:
            news_user_stance_map[news] = {}
            for i, stance_map in enumerate(all_stance_map):
                if news in stance_map:
                    engaging_users = stance_map[news]
                    for user, num_stances, ts_mean, ts_std in engaging_users:
                        if user not in news_user_stance_map[news]:
                            news_user_stance_map[news][user] = [list(np.zeros(self.n_stances)), 0, 0]
                        news_user_stance_map[news][user][0][i] = num_stances
                        news_user_stance_map[news][user][1] = ts_mean
                        news_user_stance_map[news][user][2] = ts_std

        for news in news_user_stance_map.keys():
            for user in news_user_stance_map[news]:
                self.stance_lists[news].append((user, news_user_stance_map[news][user]))
                self.stance_lists[user].append((news, news_user_stance_map[news][user]))
                self.edge_list.append((user, news, news_user_stance_map[news][user]))

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        return self.entities[index]

class FangTrainDataSet(Dataset):
    """
    DataSet used for train, validate and test
    """
    def __init__(self, data_batch: list, label: list):
        super().__init__()
        self.data_batch = data_batch
        self.label = label

    def __len__(self):
        return len(self.data_batch)
    def __getitem__(self, idx):
        item = {'data': self.data_batch[idx],
                'label': self.label[idx]}
        return item

