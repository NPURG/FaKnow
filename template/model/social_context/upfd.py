from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from template.model.social_context.base_gnn import _BaseGNN
from template.model.social_context.gcnfn import _BaseGCNFN

"""
User Preference-aware Fake News Detection
paper: https://arxiv.org/abs/2104.12259
code: https://github.com/safe-graph/GNN-FakeNews
"""


class UPFDGCN(_BaseGNN):
    def __init__(self, feature_size: int, hidden_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
        """
        super().__init__(feature_size, hidden_size, True)
        self.conv = GCNConv(self.feature_size, self.hidden_size)


class UPFDSAGE(_BaseGNN):
    def __init__(self, feature_size: int, hidden_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
        """
        super().__init__(feature_size, hidden_size, True)
        self.conv = SAGEConv(self.feature_size, self.hidden_size)


class UPFDGAT(_BaseGNN):
    def __init__(self, feature_size: int, hidden_size=128):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): Default=128
        """
        super().__init__(feature_size, hidden_size, True)
        self.conv = GATConv(self.feature_size, self.hidden_size)


class UPFDGCNFN(_BaseGCNFN):
    def __init__(self,
                 feature_size: int,
                 hidden_size=128,
                 dropout_ratio=0.5):
        """

            Args:
                feature_size (int): dimension of input node feature
                hidden_size (int): dimension of hidden layer. Default=128
                dropout_ratio (float): dropout ratio. Default=0.5
            """
        super(UPFDGCNFN, self).__init__(feature_size, hidden_size, dropout_ratio, True)
