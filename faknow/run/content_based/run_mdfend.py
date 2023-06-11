from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from faknow.data.dataset.text import TextDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.mdfend import MDFEND
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


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


def run_mdfend(train_path: str,
               bert='hfl/chinese-roberta-wwm-ext',
               max_len=170,
               domain_num=9,
               batch_size=64,
               num_epochs=50,
               lr=0.0005,
               weight_decay=5e-5,
               step_size=100,
               gamma=0.98,
               metrics: List = None,
               validate_path: str = None,
               test_path: str = None):
    tokenizer = MDFENDTokenizer(max_len, bert)
    train_set = TextDataset(train_path, ['text'], tokenizer)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    if validate_path is not None:
        validate_set = TextDataset(validate_path, ['text'], tokenizer)
        val_loader = DataLoader(validate_set, batch_size, shuffle=False)
    else:
        val_loader = None

    model = MDFEND(bert, domain_num)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    evaluator = Evaluator(metrics)

    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = TextDataset(test_path, ['text'], tokenizer)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print('test result: ', dict2str(test_result))


def main():
    path = "F:\\dataset\\weibo21\\all.json"
    run_mdfend(path)


if __name__ == '__main__':
    main()