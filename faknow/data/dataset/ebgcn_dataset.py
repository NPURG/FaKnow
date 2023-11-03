import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data


class EBGCNDataset(Dataset):
    def __init__(self, nodes_index: list, treeDic: dict, data_path: str, lower=2, upper=100000,
                 tddroprate=0, budroprate=0):
        """
        nodes_index(list): node index list.
        treeDic(dict): dictionary of graph.
        data_path(str): the path of data doc.
        lower(int): the minimum of graph size. default=2.
        upper(int): the maximum of graph size. default=100000.
        tddroprate(float): the dropout rate of TDgraph.
        budroprate(float): the dropout rate of BUgraph.
        """
        self.nodes_index = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, nodes_index))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.nodes_index)

    def __getitem__(self, index):
        id = self.nodes_index[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]),
                    label=torch.LongTensor([int(data['y'])]),
                    root=torch.LongTensor(data['root']),
                    root_index=torch.LongTensor([int(data['rootindex'])]))
