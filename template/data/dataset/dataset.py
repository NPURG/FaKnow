import torch.utils.data
from torch.utils.data.dataset import T_co


# including data processing, downloading some open datasets and some custom configuration
class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def __getitem__(self, index) -> T_co:
        pass
