import torch

import random
import numpy as np


class DropEdge:
    def __init__(self, td_drop_rate, bu_drop_rate):
        """
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
        self.td_drop_rate = td_drop_rate
        self.bu_drop_rate = bu_drop_rate

    def __call__(self, data):
        edge_index = data.edge_index

        if self.td_drop_rate > 0:
            row = list(edge_index[0])
            col = list(edge_index[1])
            length = len(row)
            poslist = random.sample(range(length),
                                    int(length * (1 - self.td_drop_rate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edge_index = [row, col]
        else:
            new_edge_index = edge_index

        bu_row = list(edge_index[1])
        bu_col = list(edge_index[0])
        if self.bu_drop_rate > 0:
            length = len(bu_row)
            poslist = random.sample(range(length),
                                    int(length * (1 - self.bu_drop_rate)))
            poslist = sorted(poslist)
            row = list(np.array(bu_row)[poslist])
            col = list(np.array(bu_col)[poslist])
            bu_new_edge_index = [row, col]
        else:
            bu_new_edge_index = [bu_row, bu_col]

        data.edge_index = torch.LongTensor(new_edge_index)
        data.BU_edge_index = torch.LongTensor(bu_new_edge_index)
        data.root = torch.FloatTensor(data.x[0])
        data.root_index = torch.LongTensor([0])

        return data
