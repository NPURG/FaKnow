import pickle
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from transformers import BertTokenizer

from template.data.dataset.mdfend_dataset import JsonDataset
from template.data.dataset.text_dataset import TensorTextDataset
from template.evaluate.evaluator import Evaluator
from template.model.multi_modal.mdfend import MDFEND
from template.train.trainer import BaseTrainer


def get_df(name: str) -> pd.DataFrame:
    with open(f"E:\\dataset\\weibo21_sub_df\\{name}.pkl", "rb") as f:
        df = pickle.load(f)
        df = df[df['category'] != '无法确定']
    return df


def bert_tokenize(max_length: int,
                  texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    inputs = tokenizer(texts,
                       return_tensors='pt',
                       max_length=max_length,
                       add_special_tokens=True,
                       padding='max_length',
                       truncation=True)
    return inputs['input_ids'], inputs['attention_mask']


def load_dataset(name: str):
    df = get_df(name)
    category_dict = {
        "科技": 0,
        "军事": 1,
        "教育考试": 2,
        "灾难事故": 3,
        "政治": 4,
        "医药健康": 5,
        "财经商业": 6,
        "文体娱乐": 7,
        "社会生活": 8
    }
    content = df['content'].tolist()
    token_ids, masks = bert_tokenize(170, content)
    label = torch.tensor(df['label'].astype(int).to_numpy())
    category = torch.tensor(
        df['category'].apply(lambda c: category_dict[c]).to_numpy())
    dataset = TensorTextDataset(texts=token_ids,
                                labels=label,
                                mask=masks,
                                domain=category)
    return dataset


def read_txt(path: str, other_params: Dict[str, Any]):
    max_length = other_params['max_length']
    with open(path, encoding='utf-8') as f:
        domain = f.readline()
        text = f.readline()  # 第二行才是文本
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, add_special_tokens=True,
                       padding='max_length', truncation=True)
    return int(domain), inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()


def run_mdfend(root: str):
    model = MDFEND(
                   'hfl/chinese-roberta-wwm-ext',
                   mlp_dims=[384],
                   dropout_rate=0.2,
                   domain_num=9)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.0005,
                                 weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=100,
                                                gamma=0.98)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    # train_set, test_set, val_set = load_dataset('train'), load_dataset(
    #     'test'), load_dataset('val')
    dataset = JsonDataset(root)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [1400, 200, 400])

    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_set,
                validate_data=val_set,
                batch_size=128,
                epochs=50,
                saved=True)
    test_result = trainer.evaluate(test_set, batch_size=64)
    print(test_result)


if __name__ == '__main__':
    path = "F:\\dataset\\weibo21\\all.json"
    run_mdfend(path)
