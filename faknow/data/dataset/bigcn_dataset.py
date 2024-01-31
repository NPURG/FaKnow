import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from faknow.data.process.process import DropEdge


class BiGCNDataset(Dataset):
    """
    Dataset for BiGCN.
    """
    def __init__(self,
                 nodes_index: List,
                 tree_dict: Dict,
                 data_path: str,
                 lower=2,
                 upper=100000,
                 td_drop_rate=0.2,
                 bu_drop_rate=0.2):
        """
        Args:
            nodes_index(List): node index list.
            tree_dict(Dict): dictionary of graph.
            data_path(str): the path of data doc, where each sample is a graph
                with node features,  edge indices, the label and the root
                saved in npz file.
            lower(int): the minimum of graph size. default=2.
            upper(int): the maximum of graph size. default=100000.
            td_drop_rate(float): the dropout rate of TDgraph, default=0.2
            bu_drop_rate(float): the dropout rate of BUgraph, default=0.2
        """
        self.nodes_index = list(
            filter(
                lambda id_: id_ in tree_dict and lower <= len(tree_dict[id_])
                <= upper, nodes_index))
        self.treeDic = tree_dict
        self.data_path = data_path
        self.drop_edge = DropEdge(td_drop_rate, bu_drop_rate)

    def __len__(self):
        return len(self.nodes_index)

    def __getitem__(self, index):
        """
        Args:
            index (int): item index

        Returns:
            Data: pyg Data with features named x, y, edge_index,
                BU_edge_index, root and root_index
        """
        id_ = self.nodes_index[index]
        data = np.load(os.path.join(self.data_path, id_ + ".npz"),
                       allow_pickle=True)

        graph_data = Data(x=data['x'], edge_index=data['edge_index'])
        graph_data = self.drop_edge(graph_data)

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=graph_data.edge_index,
                    BU_edge_index=graph_data.BU_edge_index,
                    y=torch.LongTensor([int(data['y'])]),
                    root=torch.LongTensor(data['root']),
                    root_index=torch.LongTensor([int(data['root_index'])]))
