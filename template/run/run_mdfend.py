from typing import List, Dict

import torch
from torch.utils.data import random_split, DataLoader
from transformers import BertTokenizer

from data.dataset.text import TextDataset
from template.evaluate.evaluator import Evaluator
from template.model.content_based.mdfend import MDFEND
from template.train.trainer import BaseTrainer
from utils.util import dict2str


class MDFENDTokenizer:
    def __init__(self, max_len=170, bert="hfl/chinese-roberta-wwm-ext"):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert)

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(texts,
                                return_tensors='pt',
                                max_length=self.max_len,
                                add_special_tokens=True,
                                padding='max_length',
                                truncation=True)
        return {'token_id': inputs['input_ids'], 'mask': inputs['attention_mask']}


def run_mdfend(path: str):
    tokenizer = MDFENDTokenizer()
    dataset = TextDataset(path, ['text'], tokenizer)

    validate_size = int(len(dataset) * 0.1)
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - validate_size - test_size
    train_set, validate_set, test_set = random_split(dataset, [train_size, validate_size, test_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(validate_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = MDFEND('hfl/chinese-roberta-wwm-ext', 9)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.0005,
                                 weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=100,
                                                gamma=0.98)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epoch=50, validate_loader=val_loader)
    test_result = trainer.evaluate(test_loader)
    print('test result: ', dict2str(test_result))


def main():
    path = "F:\\dataset\\weibo21\\all.json"
    run_mdfend(path)


if __name__ == '__main__':
    main()
