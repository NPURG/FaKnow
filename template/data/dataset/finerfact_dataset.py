from typing import Optional

from torch import Tensor
from torch.utils.data import Dataset


class FinerFactDataset(Dataset):
    def __init__(self, token_id: Tensor, mask: Tensor, type_id: Tensor, label: Tensor, post_rank: Tensor,
                 user_rank: Tensor,
                 keyword_rank: Tensor, user_metadata: Optional[Tensor] = None):
        super().__init__()
        self.user_metadata = user_metadata
        self.keyword_rank = keyword_rank
        self.user_rank = user_rank
        self.post_rank = post_rank
        self.label = label
        self.type_id = type_id
        self.mask = mask
        self.token_id = token_id
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        item = {
            'token_id': self.token_id[index],
            'mask': self.mask[index],
            'type_id': self.type_id[index],
            'label': self.label[index],
            'post_rank': self.post_rank[index],
            'user_rank': self.user_rank[index],
            'keyword_rank': self.keyword_rank[index]
        }
        if self.user_metadata is not None:
            item['user_metadata'] = self.user_metadata[index]
        return item
