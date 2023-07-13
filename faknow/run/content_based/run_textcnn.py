import pickle
import re
from typing import Dict, List, Any, Optional, Callable

import jieba
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from faknow.data.dataset.text import TextDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.textcnn import TextCNN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str, read_stop_words

__all__ = ['TokenizerTextCNN', 'run_textcnn', 'run_textcnn_from_yaml']


class TokenizerTextCNN:
    def __init__(self, vocab: Dict[str, int], max_len=255, stop_words: List[str] = None, language='zh') -> None:
        assert language in ['zh', 'en'], "language must be one of {zh, en}"
        self.language = language
        self.vocab = vocab
        self.max_len = max_len
        if stop_words is None:
            stop_words = []
        self.stop_words = stop_words

    def __call__(self, texts: List[str]) -> torch.Tensor:
        token_ids = []
        for text in texts:
            cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", text).strip().lower()

            if self.language == 'zh':
                split_words = jieba.cut(cleaned_text)
            else:
                split_words = cleaned_text.split()
            words = [word for word in split_words if word not in self.stop_words]

            token_id = [self.vocab[word] for word in words]
            real_len = len(token_id)
            if real_len < self.max_len:
                token_id.extend([0] * (self.max_len - real_len))
            elif real_len > self.max_len:
                token_id = token_id[:self.max_len]
            token_ids.append(token_id)

        return torch.tensor(token_ids)


def run_textcnn(train_path: str,
                vocab: Dict[str, int],
                stop_words: List[str],
                word_vectors: torch.Tensor,
                language='zh',
                max_len=255,
                filter_num=100,
                kernel_sizes: List[int] = None,
                activate_func: Optional[Callable] = F.relu,
                dropout=0.5,
                freeze=False,
                batch_size=50,
                lr=0.001,
                num_epochs=25,
                metrics: List = None,
                validate_path: str = None,
                test_path: str = None,
                device='cpu') -> None:
    tokenizer = TokenizerTextCNN(vocab, max_len, stop_words, language)

    train_set = TextDataset(train_path, ['text'], tokenizer)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    if validate_path is not None:
        val_set = TextDataset(validate_path, ['text'], tokenizer)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
    else:
        val_loader = None

    model = TextCNN(word_vectors, filter_num, kernel_sizes, activate_func, dropout, freeze)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr)
    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = TextDataset(test_path, ['text'], tokenizer, ['image'])
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def _parse_kargs(config: Dict[str, Any]) -> Dict[str, Any]:
    with open(config['vocab'], 'rb') as f:
        config['vocab'] = pickle.load(f)
    with open(config['word_vectors'], 'rb') as f:
        config['word_vectors'] = pickle.load(f)
    config['stop_words'] = read_stop_words(config['stop_words'])
    return config


def run_textcnn_from_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_textcnn(**_parse_kargs(_config))
