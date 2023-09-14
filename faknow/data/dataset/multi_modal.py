import os
from typing import List, Callable, Any, Dict

import torch

from faknow.data.dataset.text import TextDataset


class MultiModalDataset(TextDataset):
    """
    Dataset for json file with post texts and images,
    allow users to tokenize texts and convert them into tensors,
    inherit from TextDataset.

    Attributes:
        root (str): absolute path to json file
        data (dict): data in json file
        feature_names (List[str]): names of all features in json file
        tokenize (Callable[[List[str]], Any]): function to tokenize text,
            which takes a list of texts and
            returns a tensor or a dict of tensors
        text_features (dict): a dict of text features, key is feature name,
            value is feature values
        image_features (List[str]): a list of image features
        transform (Callable[[str], Any]): function to transform image,
            which takes a path to image and
            returns a tensor or a dict of tensors
    """

    def __init__(self, path: str, text_features: List[str],
                 tokenize: Callable[[List[str]], Any],
                 image_features: List[str], transform: Callable[[str], Any]):
        """
        Args:
            path (str): absolute path to json file
            text_features (List[str]): a list of names of text features in json file
            tokenize (Callable[[List[str]], Any]): function to tokenize text,
                which takes a list of texts and
                returns a tensor or a dict of tensors
            image_features (List[str]): a list of names of image features in json file
            transform (Callable[[str], Any]): function to transform image,
                which takes a path to image and
                returns a tensor or a dict of tensors
        """

        super().__init__(path, text_features, tokenize, False)

        self.transform = transform
        self.image_features = []
        for name in image_features:
            self.process_image(name)

        self._to_tensor()

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Args:
            index (int): index of item to get

        Returns:
            item (dict): a dict of features of the item
        """

        item = {}
        for feature_name, feature_values in self.data.items():
            if feature_name in self.image_features:
                value = self.transform(
                    os.path.join(self.root, feature_values[index]))
                if type(value) is not torch.Tensor and type(value) is not dict:
                    raise RuntimeError(
                        'return type of transform function must be tensor')
                # todo 递归字典的情况，是否需要展开
                # self.data.update(new_text)
            elif feature_name in self.text_features and type(
                    feature_values) is dict:
                value = {k: v[index] for k, v in feature_values.items()}
            else:
                value = feature_values[index]
            item[feature_name] = value
        return item

    def process_image(self, name: str):
        """
        Mark a feature as image features.

        Args:
            name (str): name of feature to mark as image features

        Raises:
            ValueError: if there is no feature named 'name'
            ValueError: if feature has already been marked as image features
        """

        self.check_feature(name)
        if name in self.image_features:
            raise ValueError(
                f"'{name}' has already been marked as image features")
        self.image_features.append(name)

    def remove_image(self, name: str):
        """
        Remove a feature from image features.

        Args:
            name (str): name of feature to remove from image features

        Raises:
            ValueError: if there is no feature named 'name'
            ValueError: if feature has not been marked as image features
        """

        self.check_feature(name)
        if name not in self.image_features:
            raise ValueError(f"'{name}' has not been marked as image features")
        self.image_features.remove(name)

    def _to_tensor(self):
        """
        Convert all features in data into tensor,
        including text features and image features.
        """

        for name, values in self.data.items():
            if name not in self.image_features and name not in self.text_features:
                if type(values[0]) is int or type(values[0]) is float:
                    try:
                        self.data[name] = torch.tensor(values)
                    except RuntimeError as e:
                        e.args = f"fail to convert '{name}' feature into tensor, please check data type"
                        raise e
