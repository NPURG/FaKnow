import json
from typing import Callable, List

import torch
from torch.utils.data import Dataset


class JsonDataset(Dataset):
    def __init__(self, path: str, transform: Callable[[str], torch.Tensor],
                 tokenize: Callable[[List[str]], torch.Tensor], **kargs):
        super().__init__(**kargs)
        self.transform = transform
        self.tokenize = tokenize

        self.images = []
        self.texts = []
        labels = []
        post_ids = []

        with open(path, encoding='utf-8') as f:
            samples = json.load(f)
            posts = samples[:]

        for post in posts:
            labels.append(post['label'])
            self.texts.append(post['text'])
            post_ids.append(post['post_id'])
            self.images.append(post['image'])

        self.token_ids = self.tokenize(self.texts)
        self.labels = torch.tensor(labels)
        self.post_ids = torch.tensor(post_ids)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        return self.post_ids[index], self.token_ids[index], image, self.labels[index]

    def __len__(self):
        return len(self.labels)
