import os
from typing import List, Callable, Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset for json file with post texts,
    allow users to tokenize texts and convert them into tensors.

    Attributes:
        root (str): absolute path to json file
        data (dict): data in json file
        feature_names (List[str]): names of all features in json file
        tokenize (Callable[[List[str]], Any]): function to tokenize text,
            which takes a list of texts and returns a tensor or a dict of tensors
        text_features (dict): a dict of text features, key is feature name,
            value is feature values
    """

    def __init__(self,
                 path: str,
                 text_features: List[str],
                 tokenize: Callable[[List[str]], Any],
                 to_tensor=True):
        """
        Args:
            path (str): absolute path to json file
            text_features (List[str]): a list of names of text features in json file
            tokenize (Callable[[List[str]], Any]): function to tokenize text,
                which takes a list of texts and returns a tensor or a dict of tensors
            to_tensor (bool, optional): whether to convert all features into tensor. Default=True.
        """

        super().__init__()
        self.root = os.path.dirname(path)

        self.data = pd.read_json(path, orient='records').to_dict(orient='list')
        self.feature_names = list(self.data.keys())

        self.tokenize = tokenize
        self.text_features = {}
        for name in text_features:
            self.process_text(name)

        if to_tensor:
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
            if feature_name in self.text_features and type(
                    feature_values) is dict:
                value = {k: v[index] for k, v in feature_values.items()}
            else:
                value = feature_values[index]
            item[feature_name] = value
        return item

    def __len__(self):
        return len(self.data['label'])

    def check_feature(self, name: str):
        """
        Args:
            name (str): name of feature to check

        Raises:
            ValueError: if there is no feature named 'name'
        """

        if name not in self.feature_names:
            raise ValueError(f"there is no feature named '{name}'")

    def process_text(self, name: str):
        """
        process text feature with tokenize function,
        store the old value of the feature in text_features,
        and store the new value in data.

        Args:
            name (str): name of text feature to process
        """

        self.check_feature(name)
        if name in self.text_features:
            raise ValueError(
                f"'{name}' has already been marked as text features")
        new_text = self.tokenize(self.data[name])
        if type(new_text) is dict:
            for k, v in new_text.items():
                if type(v) is not torch.Tensor:
                    raise TypeError(
                        f"the value of '{k}' returned by tokenize must be tensor"
                    )
            # todo 递归字典的情况，是否需要展开
            # self.data.update(new_text)
        elif type(new_text) is not torch.Tensor:
            raise TypeError("return type of tokenize function must be tensor")

        self.text_features[name] = self.data[name]  # store the old value
        self.data[name] = new_text  # store the new value

    def remove_text(self, name: str):
        """
        remove text feature from self.text_features

        Args:
            name (str): name of text feature to remove

        Raises:
            ValueError: if there is no feature named 'name'
            ValueError: if 'name' has not been marked as text features
        """

        self.check_feature(name)
        if name not in self.text_features:
            raise ValueError(f"'{name}' has not been marked as text features")
        self.data[name] = self.text_features[name]  # restore old value
        del self.text_features[name]  # remove new value

    def _to_tensor(self):
        """
        convert all features in data into tensor

        Raises:
            RuntimeError: if fail to convert feature into tensor
        """

        for name, values in self.data.items():
            if name not in self.text_features:
                if type(values[0]) is int or type(values[0]) is float:
                    try:
                        self.data[name] = torch.tensor(values)
                    except RuntimeError as e:
                        e.args = f"fail to convert '{name}' feature into tensor, please check data type"
                        raise e
