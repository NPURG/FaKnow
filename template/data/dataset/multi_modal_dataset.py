from typing import Tuple, Any, Optional, Callable, Dict, List, Union, cast, Set
from collections import Iterable

import os
import torch.utils.data
import torchvision
from torchvision.datasets.folder import default_loader
from template.data.dataset.text_dataset import TextDataset
from template.data.dataset.utils import make_dataset, walker_with_images


class MultiModalDataset(torchvision.datasets.ImageFolder):
    """
    dataset for cross-modal models using text and image information
    """

    def __init__(self, img_root: str, texts, img_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None,
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Args:
             img_root:
             texts: list, numpy.ndarray or torch.Tensor,
                the order should be the same as the order in which the images are loaded
             img_transform:
             target_transform:
             loader:
             is_valid_file:
             other_params:
        """
        super().__init__(img_root, img_transform, target_transform, loader, is_valid_file)
        self.texts = texts
        self.other_params = other_params

    def __getitem__(self, index: int) -> Tuple:
        img, label = super().__getitem__(index)
        text = self.loader(self.texts[index])
        if self.other_params is None:
            return text, img, label
        return text, img, label, self.other_params

    @property
    def data(self):
        return self.texts, [self.transform(img) if self.transform is not None else img for img in self.imgs]


class TensorCrossModalDataset(torch.utils.data.Dataset):

    def __init__(self, texts: torch.Tensor, images: torch.Tensor, labels: torch.Tensor):
        if len(texts) != len(images) or len(images) != len(labels) or len(texts) != len(labels):
            raise RuntimeError('the first dimension among texts, images and labels must be the same')

        self.texts = texts
        self.images = images
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self.texts[index], self.images[index], self.labels[index]


class MultiModalDataset2(TextDataset):
    def __init__(self, root: str,
                 word_embedding: Optional[Callable] = None,
                 image_transform: Optional[Callable] = None,
                 image_loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 other_params: Optional[Dict[str, Any]] = None,
                 walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set], None]] = walker_with_images):
        super().__init__(root, word_embedding, other_params, walk_class_dir)
        self.images = [sample[1] for sample in self.samples]
        self.image_transform = image_transform
        self.image_loader = image_loader

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        text, label = super().__getitem__(index)
        image = self.image_loader(self.images[index])
        if self.image_transform is not None:
            image = self.image_transform(image)
        return text, image, label
