import os
from typing import List, Callable, Any

import torch
from PIL import Image

from template.data.dataset.text import TextDataset


class MultiModalDataset(TextDataset):
    def __init__(self, path: str, text_features: List[str],
                 tokenize: Callable[[List[str]], Any],
                 image_features: List[str], transform: Callable[[Image.Image],
                                                                torch.Tensor]):
        super().__init__(path, text_features, tokenize, False)

        self.transform = transform
        self.image_features = []
        for name in image_features:
            self.process_image(name)

        self._to_tensor()

    def __getitem__(self, index):
        item = {}
        for feature_name, feature_values in self.data.items():
            if feature_name in self.image_features:
                img = self._pil_load(
                    os.path.join(self.root, feature_values[index]))
                value = self.transform(img)
                if type(value) is not torch.Tensor:
                    raise RuntimeError(
                        'return type of transform function must be tensor')
            elif feature_name in self.text_features and type(
                    feature_values) is dict:
                value = {k: v[index] for k, v in feature_values.items()}
            else:
                value = feature_values[index]
            item[feature_name] = value
        return item

    def process_image(self, name: str):
        self.check_feature(name)
        if name in self.image_features:
            raise ValueError(
                f"'{name}' has already been marked as image features")
        self.image_features.append(name)

    def remove_image(self, name: str):
        self.check_feature(name)
        if name not in self.image_features:
            raise ValueError(f"'{name}' has not been marked as image features")
        self.image_features.remove(name)

    def _pil_load(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _to_tensor(self):
        for name, values in self.data.items():
            if name not in self.image_features and name not in self.text_features:
                if type(values[0]) is int or type(values[0]) is float:
                    try:
                        self.data[name] = torch.tensor(values)
                    except RuntimeError as e:
                        e.args = f"fail to convert '{name}' feature into tensor, please check data type"
                        raise e
