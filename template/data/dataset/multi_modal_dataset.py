from typing import Tuple, Any, Optional, Callable, Dict, List, Set

import torch.utils.data
import torchvision
from torchvision.datasets.folder import default_loader

from template.data.dataset.text_dataset import FolderTextDataset, TensorTextDataset
from template.data.dataset.utils import walker_with_images


class MultiModalDataset(torchvision.datasets.ImageFolder):

    # todo 是否需要label_transform
    def __init__(self,
                 img_root: str,
                 texts,
                 img_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 other_params: Optional[Dict[str, Any]] = None):
        super().__init__(img_root, img_transform, loader=loader)
        self.texts = texts
        self.other_params = other_params

    def __getitem__(self, index: int) -> Tuple:
        img, label = super().__getitem__(index)
        text = self.texts[index]
        if self.other_params is None:
            return text, img, label
        other_params = {k: v[index] for k, v in self.other_params.items()}
        return text, img, other_params, label

    @property
    def data(self):
        return self.texts, [
            self.transform(img) if self.transform is not None else img
            for img in self.imgs
        ]


class TensorMultiModalDataset(TensorTextDataset):
    """
    inherit from TensorTextDataset for multi-modal data
    """

    def __init__(self,
                 texts: torch.Tensor = None,
                 images: torch.Tensor = None,
                 labels: torch.Tensor = None,
                 samples: Optional[List[Tuple[torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]]] = None,
                 **other_data: torch.Tensor):
        """
        Args:
            texts:
            images:
            labels:
            samples: instead of inputting texts, image and labels, they can be combined as a list of tuples (text, image, label).
            other_data: other data the model needs, it should be torch tensor
        """
        super().__init__(texts, labels, samples, **other_data)

        if samples is not None:
            images = [sample[1] for sample in samples]
        self.images = images

        if len(self.texts) != len(self.images) or len(self.images) != len(
                self.labels) or len(self.texts) != len(self.labels):
            raise RuntimeError(
                'the first dimension among texts, images and labels must be the same'
            )

    def __getitem__(self, index: int) -> Tuple:
        item = super().__getitem__(index)
        if len(item) == 2:
            # text, image, label
            return item[0], self.images[index], item[1]
        # text, image, other_data, label
        return item[0], self.images[index], item[1], item[2]


class FolderMultiModalDataset(FolderTextDataset):
    def __init__(
            self,
            root: str,
            embedding: Callable[[str, Optional[Dict]], Any],
            transform: Optional[Callable],
            image_loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set],
                                              None]] = walker_with_images,
            embedding_params: Dict[str, Any] = None,
            **other_data: torch.Tensor):
        super().__init__(root, embedding, walk_class_dir, embedding_params, **other_data)
        self.images = [sample[1] for sample in self.samples]
        self.transform = transform
        self.image_loader = image_loader

    def __getitem__(self, index: int) -> Tuple:
        item = super().__getitem__(index)
        image = self.image_loader(self.images[index])
        image = self.transform(image)

        if len(item) == 2:
            # text, image, label
            return item[0], image, item[1]
        # text, image, other_data, label
        return item[0], image, item[1], item[2]
