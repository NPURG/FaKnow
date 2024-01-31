import os
from typing import Dict, List

import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn as nn


class _TweetNode(object):
    """
    generate node tweet graph.
    """
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


class TrustRDDataset(Dataset):
    """
    Dataset for TrustRD
    """
    def __init__(self,
                 nodes_index: List,
                 tree_dict: Dict,
                 data_path: str,
                 lower=2,
                 upper=100000,
                 drop_rate=0):
        """
        Args:
            nodes_index(List): node index list.
            tree_dict(Dict): dictionary of graph.
            data_path(str): the path of data doc, where each sample is a graph
                with node features,  edge indices, the label and the root
                saved in npz file.
            lower(int): the minimum of graph size. default=2.
            upper(int): the maximum of graph size. default=100000.
            drop_rate(float): the dropout rate of edge. default=0
        """
        self.nodes_index = list(
            filter(
                lambda id_: id_ in tree_dict and lower <= len(tree_dict[id_]) <= upper, nodes_index))
        self.treeDic = tree_dict
        self.data_path = data_path
        self.drop_rate = drop_rate
        self.mask_rate = nn.Parameter(
            torch.zeros(1))  # learnable parameter for masking nodes

    def __len__(self):
        return len(self.nodes_index)

    def __getitem__(self, index):
        id_ = self.nodes_index[index]
        data = np.load(os.path.join(self.data_path, id_ + ".npz"),
                       allow_pickle=True)

        edge_index = data['edge_index']

        # construct graph
        tree = self.treeDic[id_]
        index2node = {}
        for i in tree:
            node = _TweetNode(idx=i)
            index2node[i] = node

        for j in tree:
            indexC = j
            indexP = tree[j]['parent']
            nodeC = index2node[indexC]

            if not indexP == 'None':
                nodeP = index2node[int(indexP)]
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
            else:
                root_index = indexC - 1
        # todo drop edge
        mask = [0 for _ in range(len(index2node))]
        root_node = index2node[int(root_index + 1)]
        que = root_node.children.copy()
        while len(que) > 0:
            cur = que.pop()
            if random.random() >= self.mask_rate:
                mask[int(cur.idx) - 1] = 1
                for child in cur.children:
                    que.append(child)
        mask[root_index] = 0

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    mask=torch.tensor(mask, dtype=torch.bool),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([int(data['y'])]),
                    root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['root_index'])]))
