import pickle
import re
from typing import Dict, List, Any
import warnings

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

__all__ = [
    'TokenizerEANN', 'transform_eann', 'run_eann', 'run_eann_from_yaml'
]


class TokenizerEANN:
    """
    tokenizer for EANN
    """
    def __init__(self,
                 vocab: Dict[str, int],
                 max_len=255,
                 stop_words: List[str] = None,
                 language='zh') -> None:
        """

        Args:
            vocab (Dict[str, int]): vocabulary of the corpus
            max_len (int): max length of the text, default=255
            stop_words (List[str]): stop words, default=None
            language (str): language of the corpus, 'zh' or 'en', default='zh'
        """

        assert language in ['zh', 'en'], "language must be one of {zh, en}"
        self.language = language
        self.vocab = vocab
        self.max_len = max_len
        if stop_words is None:
            stop_words = []
        self.stop_words = stop_words

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        tokenize texts

        Args:
            texts (List[str]): texts to be tokenized

        Returns:
            Dict[str, torch.Tensor]: tokenized texts with key 'token_id' and 'mask'
        """

        token_ids = []
        masks = []
        for text in texts:
            cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]",
                                  "", text).strip().lower()

            if self.language == 'zh':
                split_words = jieba.cut_for_search(cleaned_text)
                words = " ".join([
                    word for word in split_words if word not in self.stop_words
                ])
            else:
                words = [
                    word for word in cleaned_text.split()
                    if word not in self.stop_words
                ]

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

        return {
            'token_id': torch.tensor(token_ids),
            'mask': torch.stack(masks)
        }


def transform_eann(path: str) -> torch.Tensor:
    """
    transform image to tensor for EANN

    Args:
        path (str): image path

    Returns:
        torch.Tensor: tensor of the image, shape=(3, 224, 224)
    """

    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(img)


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
    run EANN, including training, validation and testing.
    If validate_path and test_path are None, only training is performed.

    Args:
        train_path (str): path of the training set
        vocab (Dict[str, int]): vocabulary of the corpus
        stop_words (List[str]): stop words
        word_vectors (torch.Tensor): word vectors
        language (str): language of the corpus, 'zh' or 'en', default='zh'
        max_len (int): max length of the text, default=255
        batch_size (int): batch size, default=100
        event_num (int): number of events, default=None
        lr (float): learning rate, default=0.001
        num_epochs (int): number of epochs, default=100
        metrics (List): metrics, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        validate_path (str): path of the validation set, default=None
        test_path (str): path of the test set, default=None
        device (str): device, default='cpu'
    """

    tokenizer = TokenizerEANN(vocab, max_len, stop_words, language)

    # todo 是否在函数中加入validation_size，允许从train_path中划分出一部分作为validation set
    train_set = MultiModalDataset(train_path, ['text'], tokenizer, ['image'],
                                  transform_eann)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    if event_num is None:
        event_num = torch.max(train_set.data['domain']).item() + 1
        warnings.warn(f"event_num is not specified,"
                      f"use max domain number in training set + 1: {event_num} as event_num")

    if validate_path is not None:
        val_set = MultiModalDataset(validate_path, ['text'], tokenizer,
                                    ['image'], transform_eann)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
    else:
        val_loader = None

    model = EANN(event_num, embed_weight=word_vectors)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters())), lr)
    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model,
                          evaluator,
                          optimizer,
                          device=device)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = MultiModalDataset(test_path, ['text'], tokenizer, ['image'],
                                     transform_eann)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def _parse_kargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    parse kargs from config dict

    Args:
        config (Dict[str, Any]): config dict, keys are the same as the args of `run_eann`

    Returns:
        Dict[str, Any]: converted kargs
    """

    with open(config['vocab'], 'rb') as f:
        config['vocab'] = pickle.load(f)
    with open(config['word_vectors'], 'rb') as f:
        config['word_vectors'] = pickle.load(f)
    config['stop_words'] = read_stop_words(config['stop_words'])
    return config


def run_eann_from_yaml(path: str) -> None:
    """
    run EANN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_eann(**_parse_kargs(_config))
