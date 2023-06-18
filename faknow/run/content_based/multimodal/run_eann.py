import pickle
import re
from typing import Dict, List, Any

import jieba
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.eann import EANN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str, read_stop_words

__all__ = ['TokenizerEANN', 'transform_eann', 'adjust_lr_eann', 'run_eann', 'run_eann_from_yaml']


class TokenizerEANN:
    def __init__(self, vocab: Dict[str, int], max_len=255, stop_words: List[str] = None, language='zh') -> None:
        assert language in ['zh', 'en'], "language must be one of {zh, en}"
        self.language = language
        self.vocab = vocab
        self.max_len = max_len
        if stop_words is None:
            stop_words = []
        self.stop_words = stop_words

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        token_ids = []
        masks = []
        for text in texts:
            cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", text).strip().lower()

            if self.language == 'zh':
                split_words = jieba.cut_for_search(cleaned_text)
                words = " ".join([word for word in split_words if word not in self.stop_words])
            else:
                words = [word for word in cleaned_text.split() if word not in self.stop_words]

            token_id = [self.vocab[word] for word in words]
            real_len = len(token_id)
            if real_len < self.max_len:
                token_id.extend([0] * (self.max_len - real_len))
            elif real_len > self.max_len:
                token_id = token_id[:self.max_len]
            token_ids.append(token_id)

            mask = torch.zeros(self.max_len)
            mask[:real_len] = 1
            masks.append(mask)

        return {'token_id': torch.tensor(token_ids), 'mask': torch.stack(masks)}


def transform_eann(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(img)


def adjust_lr_eann(epoch: int) -> float:
    return 0.001 / (1. + 10 * (float(epoch) / 100)) ** 0.75


def run_eann(train_path: str,
             vocab: Dict[str, int],
             stop_words: List[str],
             word_vectors: torch.Tensor,
             language='zh',
             max_len=255,
             batch_size=100,
             event_num: int = None,
             lr=0.001,
             num_epochs=100,
             metrics: List = None,
             validate_path: str = None,
             test_path: str = None,
             device='cpu') -> None:
    """
    todo run函数是否不要加过多参数，直接最基础最原始的几个参数就可以了，不需要很复杂，对于复杂功能就让用户自己写代码
    直接传入数据
    """

    tokenizer = TokenizerEANN(vocab, max_len, stop_words, language)

    # todo 是否在函数中加入validation_size，允许从train_path中划分出一部分作为validation set
    train_set = MultiModalDataset(train_path, ['text'], tokenizer, ['image'], transform_eann)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    if event_num is None:
        event_num = torch.max(train_set.data['domain']).item() + 1

    if validate_path is not None:
        val_set = MultiModalDataset(validate_path, ['text'], tokenizer, ['image'], transform_eann)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
    else:
        val_loader = None

    model = EANN(event_num, embed_weight=word_vectors)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=adjust_lr_eann)
    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = MultiModalDataset(test_path, ['text'], tokenizer, ['image'], transform_eann)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def run_eann_from_yaml(config: Dict[str, Any]):
    with open(config['vocab'], 'rb') as f:
        config['vocab'] = pickle.load(f)
    with open(config['word_vectors'], 'rb') as f:
        config['word_vectors'] = pickle.load(f)
    config['stop_words'] = read_stop_words(config['stop_words'])
    run_eann(**config)


if __name__ == '__main__':
    with open(r'..\..\..\properties\eann.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_eann_from_yaml(_config)
