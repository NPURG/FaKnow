import json
import random

import torch
from transformers import BertTokenizer


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, **kars):
        super().__init__(**kars)
        with open(path, encoding='utf-8') as f:
            samples = json.load(f)
            random.shuffle(samples)
            posts = samples[:2000]
        
        self.domains = [post['domain'] for post in posts]
        self.labels = [post['label'] for post in posts]

        tokenizer = BertTokenizer.from_pretrained(
            "hfl/chinese-roberta-wwm-ext")
        texts = [post['text'] for post in posts]
        inputs = tokenizer(texts,
                           return_tensors='pt',
                           max_length=170,
                           add_special_tokens=True,
                           padding='max_length',
                           truncation=True)
        self.token_ids, self.masks = inputs['input_ids'], inputs[
            'attention_mask']

    def __getitem__(self, index):
        return self.token_ids[index], self.masks[index], self.domains[
            index], self.labels[index]

    def __len__(self):
        return len(self.labels)