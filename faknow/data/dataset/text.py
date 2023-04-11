import os
from typing import List, Callable, Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self,
                 path: str,
                 text_features: List[str],
                 tokenize: Callable[[List[str]], Any],
                 to_tensor=True):
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

    def __getitem__(self, index):
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
        if name not in self.feature_names:
            raise ValueError(f"there is no feature named '{name}'")

    def process_text(self, name: str):
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
        elif type(new_text) is not torch.Tensor:
            raise TypeError("return type of tokenize function must be tensor")

        self.text_features[name] = self.data[name]  # 把旧值存储
        self.data[name] = new_text  # 更改为tokenize后的新值

    def remove_text(self, name: str):
        self.check_feature(name)
        if name not in self.text_features:
            raise ValueError(f"'{name}' has not been marked as text features")
        self.data[name] = self.text_features[name]  # 恢复旧值
        del self.text_features[name]  # 删除新值

    def _to_tensor(self):
        for name, values in self.data.items():
            if name not in self.text_features:
                if type(values[0]) is int or type(values[0]) is float:
                    try:
                        self.data[name] = torch.tensor(values)
                    except RuntimeError as e:
                        e.args = f"fail to convert '{name}' feature into tensor, please check data type"
                        raise e
