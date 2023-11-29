import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn as nn


class Node_tweet(object):
    """
    generate node tweet graph.
    """

    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


class TRUSTRDDataset(Dataset):
    def __init__(self,
                 nodes_index: list,
                 treeDic: dict,
                 data_path: str,
                 lower=2,
                 upper=100000,
                 droprate=0):
        """
        Args:
            nodes_index(list): node index list.
            treeDic(dict): dictionary of graph.
            data_path(str): the path of data doc.
            lower(int): the minimum of graph size. default=2.
            upper(int): the maximum of graph size. default=100000.
            droprate(float): the dropout rate of edge. default=0
        """
        self.nodes_index = list(
            filter(
                lambda id: id in treeDic and len(treeDic[id]) >= lower and len(
                    treeDic[id]) <= upper, nodes_index))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate
        self.mask_rate = nn.Parameter(torch.zeros(1))  # learnable parameter for masking nodes

    def __len__(self):
        return len(self.nodes_index)

    def __getitem__(self, index):
        id = self.nodes_index[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"),
                       allow_pickle=True)

        edgeindex = data['edge_index']

        tree = self.treeDic[id]
        index2node = {}
        for i in tree:
            node = Node_tweet(idx=i)
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
                rootindex = indexC - 1

        mask = [0 for _ in range(len(index2node))]
        root_node = index2node[int(rootindex + 1)]
        que = root_node.children.copy()
        while len(que) > 0:
            cur = que.pop()
            if random.random() >= self.mask_rate:
                mask[int(cur.idx) - 1] = 1
                for child in cur.children:
                    que.append(child)
        mask[rootindex] = 0

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    mask=torch.tensor(mask, dtype=torch.bool),
                    edge_index=torch.LongTensor(edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['root_index'])]))
