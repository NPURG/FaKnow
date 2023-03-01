from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, Any

import numpy as np
import torch.utils.data
from nltk.tokenize import sent_tokenize

from template.data.dataset.text_dataset import FolderTextDataset


class SAFENumpyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
    ):
        super().__init__()

        x_heads = np.load(root_dir / "case_headline.npy", allow_pickle=True)
        x_bodies = np.load(root_dir / "case_body.npy", allow_pickle=True)
        x_images = np.load(root_dir / "case_image.npy", allow_pickle=True)
        # todo y应该是n*1，而非n*2
        # 代码不用动，要把 case_y_fn.npy 文件改成n*1
        y = np.load(root_dir / "case_y_fn_dim1.npy")

        self.x_heads = x_heads.astype(np.float32)
        self.x_bodies = x_bodies.astype(np.float32)
        self.x_images = x_images.astype(np.float32)
        self.y = y.astype(np.float32)

        assert self.x_heads.shape[0] == self.x_bodies.shape[0] == self.x_images.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.x_heads.shape[0]

    def __getitem__(self, index: int):
        return self.x_heads[index], self.x_bodies[index], self.x_images[index], self.y[index]


class SAFEDataset(FolderTextDataset):
    def __init__(
            self,
            root: str,
            embedding: Callable[[str, Optional[Dict]], Any],
            max_len: int = 16,
    ):
        super().__init__(root, embedding=embedding)

        self.max_len = max_len

    def _embedding(self, text: str) -> np.ndarray:
        sentences = sent_tokenize(text)
        embd = self.embedding(sentences)
        return embd

    def _padding(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] >= self.max_len:
            return x[:self.max_len]
        else:
            padded = np.zeros((self.max_len, *x.shape[1:]), dtype=x.dtype)
            padded[:x.shape[0]] = x
            return padded

    def __getitem__(self, index) -> Tuple:
        text_path, label = self.samples[index]
        with open(text_path, encoding='utf-8') as f:
            text = f.read()
        headline, body, image = text.split("\n")

        headline = self._padding(self._embedding(headline.strip())).astype(np.float32)
        body = self._padding(self._embedding(body.strip())).astype(np.float32)
        image = self._padding(self._embedding(image.strip())).astype(np.float32)

        return headline, body, image, label
