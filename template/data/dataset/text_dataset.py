import torch
import torch.utils.data
from torchvision.datasets.folder import find_classes
from typing import Optional, Callable, Dict, List, Tuple, Any, Set
from template.data.dataset.utils import make_dataset, default_walker


class TensorTextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 texts: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 samples: Optional[List[Tuple[torch.Tensor,
                                              torch.Tensor]]] = None,
                 other_data: Optional[Dict[str, torch.Tensor]] = None):
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
    def __init__(
        self,
        root: str,
        embedding: Optional[Callable[[str], Any]],
        other_data: Optional[Dict[str, torch.Tensor]] = None,
        walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set],
                                          None]] = default_walker):
        self.classes, self.class_to_idx = find_classes(root)
        self.samples = make_dataset(root, self.class_to_idx, None, None,
                                    walk_class_dir)

        self.texts = [sample[0] for sample in self.samples]
        self.labels = [sample[-1] for sample in self.samples]
        self.embedding = embedding
        self.other_data = other_data

    def __getitem__(self, index) -> Tuple:
        text = self.samples[index][0]
        label = self.samples[index][-1]
        if self.embedding is not None:
            text = self.embedding(text)

        if self.other_data is None:
            return text, label
        other_data = {k: v[index] for k, v in self.other_data.items()}
        return text, label, other_data

    def __len__(self) -> int:
        return len(self.samples)
