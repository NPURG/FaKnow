import os
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
from torch_geometric.data import Data
from faknow.utils.util import DropEdge


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
        self.dropedge = DropEdge(tddroprate, budroprate)
    def __len__(self):
        return len(self.nodes_index)

    def __getitem__(self, index):
        id = self.nodes_index[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)

        graph_data = Data(x=data['x'],
                          edge_index=data['edge_index'])
        graph_data = self.dropedge(graph_data)


        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=graph_data.edge_index,
                    BU_edge_index=graph_data.BU_edge_index,
                    y=torch.LongTensor([int(data['y'])]),
                    root=torch.LongTensor(data['root']),
                    root_index=torch.LongTensor([int(data['root_index'])]))
