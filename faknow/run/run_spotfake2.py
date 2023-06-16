import random
import re
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.spotfake import SpotFake
from faknow.train.trainer import BaseTrainer


def text_preprocessing(text):
    """
    - 删除实体@符号(如。“@united”)
    — 纠正错误(如:'&amp;' '&')
    @参数 text (str):要处理的字符串
    @返回 text (Str):已处理的字符串
    """
    # 去除 '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    #  替换'&amp;'成'&'
    text = re.sub(r'&amp;', '&', text)

    # 删除尾随空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class SpotFakeTokenizer:
    def __init__(self, tokenizer, max_len):
        self.tokenizer_bert = tokenizer
        self.max_len = max_len

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        # 定义列表存储文本处理后的结果
        input_ids_ls = []
        attention_mask_ls = []

        for text in texts:
            encoded_sent = self.tokenizer_bert.encode_plus(
                text=text_preprocessing(text),  # 预处理
                add_special_tokens=True,  # `[CLS]`&`[SEP]`
                max_length=self.max_len,  # 截断/填充的最大长度
                padding='max_length',  # 句子填充最大长度
                # return_tensors='pt',          # 返回tensor
                return_attention_mask=True,  # 返回attention mask
                truncation=True
            )
            input_ids = encoded_sent.get('input_ids')
            attention_mask = encoded_sent.get('attention_mask')

            # 转换tensor
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            # 添加到列表中去
            input_ids_ls.append(input_ids)
            attention_mask_ls.append(attention_mask)

        return {'input_ids': torch.stack(input_ids_ls), 'attention_mask': torch.stack(attention_mask_ls)}


def transform(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(img)



def run_spotfake(
        rank,
        root: str,
        pre_trained_bert_name="bert-base-uncased",
        batch_size=8,
        epochs=50,
        MAX_LEN=500,
):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    tokenizer = BertTokenizer.from_pretrained(pre_trained_bert_name, do_lower_case=True)
    tokenizer = SpotFakeTokenizer(tokenizer, MAX_LEN)

    train_path = root + "train_posts_clean.json"
    validation_path = root + "test_posts.json"
    training_set = MultiModalDataset(train_path, ['post_text'], tokenizer, ['image_id'], transform)
    validation_set = MultiModalDataset(validation_path, ['post_text'], tokenizer, ['image_id'], transform)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    model = SpotFake(pre_trained_bert_name=pre_trained_bert_name)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-5,
        eps=1e-8
    )

    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = BaseTrainer(model, evaluator, optimizer, device='cuda:0')
    trainer.fit(train_loader, epochs, validation_loader)


def main():
    root = "/root/FaKnow/dataset/example/SpotFake/twitter/"
    pre_trained_bert_name = "bert-base-uncased"
    run_spotfake(0, root, pre_trained_bert_name)


if __name__ == '__main__':
    main()