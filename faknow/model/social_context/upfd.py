from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from faknow.model.social_context.base_gnn import _BaseGNN
from faknow.model.social_context.gcnfn import _BaseGCNFN


class UPFDGCN(_BaseGNN):
    """
    User Preference-aware Fake News Detection, SIGIR 2021
    paper: https://dl.acm.org/doi/abs/10.1145/3404835.3462990
    code: https://github.com/safe-graph/GNN-FakeNews

    UPFD model with GCN layer
    """

    def __init__(self, feature_size: int, hidden_size=128):
        """
        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
        """

        super().__init__(feature_size, hidden_size, True)
        self.conv = GCNConv(self.feature_size, self.hidden_size)


class UPFDSAGE(_BaseGNN):
    """
    User Preference-aware Fake News Detection, SIGIR 2021
    paper: https://dl.acm.org/doi/abs/10.1145/3404835.3462990
    code: https://github.com/safe-graph/GNN-FakeNews

    UPFD model with SAGE layer
    """

    def __init__(self, feature_size: int, hidden_size=128):
        """
        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
        """

        super().__init__(feature_size, hidden_size, True)
        self.conv = SAGEConv(self.feature_size, self.hidden_size)


class UPFDGAT(_BaseGNN):
    """
    User Preference-aware Fake News Detection, SIGIR 2021
    paper: https://dl.acm.org/doi/abs/10.1145/3404835.3462990
    code: https://github.com/safe-graph/GNN-FakeNews

    UPFD model with GAT layer
    """

    def __init__(self, feature_size: int, hidden_size=128):
        """
        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
        """

        super().__init__(feature_size, hidden_size, True)
        self.conv = GATConv(self.feature_size, self.hidden_size)


class UPFDGCNFN(_BaseGCNFN):
    """
    User Preference-aware Fake News Detection, SIGIR 2021
    paper: https://dl.acm.org/doi/abs/10.1145/3404835.3462990
    code: https://github.com/safe-graph/GNN-FakeNews

    UPFD model with GCNFN layer
    """

    def __init__(self, feature_size: int, hidden_size=128, dropout_ratio=0.5):
        """

        Args:
            feature_size (int): dimension of input node feature
            hidden_size (int): dimension of hidden layer. Default=128
            dropout_ratio (float): dropout ratio. Default=0.5
        """

        super(UPFDGCNFN, self).__init__(feature_size, hidden_size,
                                        dropout_ratio, True)
