from typing import Optional, Callable, Dict, List, Tuple, Any, Set

import torch.utils.data
from torchvision.datasets.folder import find_classes

from template.data.dataset.utils import make_dataset, default_walker


class TensorTextDataset(torch.utils.data.Dataset):
    """
    the usage of TensorTextDataset is similar to :class:`torch.utils.data.dataset.TensorDataset`
    """

    def __init__(self,
                 texts: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 samples: Optional[List[Tuple[torch.Tensor,
                                              torch.Tensor]]] = None,
                 **other_data: torch.Tensor):
        if samples is not None:
            texts = [sample[0] for sample in samples]
            labels = [sample[-1] for sample in samples]
        if len(texts) != len(labels):
            raise RuntimeError(
                'the first dimension between texts and labels must be the same'
            )

        self.texts = texts
        self.labels = labels
        self.other_data = other_data

    def __getitem__(self, index: int) -> Tuple:
        text = self.texts[index]
        label = self.labels[index]
        if self.other_data is None:
            return text, label
        other_data = {k: v[index] for k, v in self.other_data.items()}
        return text, label, other_data

    def __len__(self) -> int:
        return len(self.labels)


class FolderTextDataset(torch.utils.data.Dataset):
    """
    the usage of FolderTextDataset is similar to :class:`torchvision.datasets.ImageFolder`
    the folder should be followings by default: ::
        root/fake/news1.txt
        root/fake/news2.txt
        ...
        root/real/news3.txt
        root/real/news4.txt
    """

    def __init__(
            self,
            root: str,
            embedding: Callable[[str, Optional[Dict]], Any],
            walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set],
                                              None]] = default_walker,
            embedding_params: Dict[str, Any] = None,
            **other_data: torch.Tensor):
        self.classes, self.class_to_idx = find_classes(root)
        self.samples = make_dataset(root, self.class_to_idx, None, None,
                                    walk_class_dir)

        self.texts = [sample[0] for sample in self.samples]
        self.labels = [sample[-1] for sample in self.samples]
        self.embedding = embedding
        self.embedding_params = embedding_params
        self.other_data = other_data

    def __getitem__(self, index) -> Tuple:
        text = self.samples[index][0]
        label = self.samples[index][-1]
        if self.embedding_params is None:
            text = self.embedding(text)
        else:
            text = self.embedding(text, self.embedding_params)

        if self.other_data is None:
            return text, label
        other_data = {k: v[index] for k, v in self.other_data.items()}
        return text, label, other_data

    def __len__(self) -> int:
        return len(self.samples)
