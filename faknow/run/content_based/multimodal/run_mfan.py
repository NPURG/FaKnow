import json
import pickle
import re
from typing import List, Dict, Any

import jieba
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.mfan import MFAN
from faknow.train.pgd_trainer import MFANTrainer
from faknow.utils.util import dict2str

__all__ = [
    'TokenizerMFAN', 'transform_mfan', 'load_adj_matrix_mfan', 'run_mfan',
    'run_mfan_from_yaml'
]


class TokenizerMFAN:
    """
    Tokenizer for MFAN
    """
    def __init__(self,
                 vocab: Dict[str, int],
                 max_len=50,
                 stop_words: List[str] = None,
                 language='zh') -> None:
        """
        Args:
            vocab (Dict[str, int]): vocabulary dict
            max_len (int): max length of text, default=50
            stop_words (List[str]): stop words list, default=None
            language (str): language of text, 'zh' or 'en', default='zh'
        """

        self.vocab = vocab
        self.max_len = max_len
        self.language = language
        if stop_words is None:
            stop_words = []
        self.stop_words = stop_words

    def __call__(self, texts: List[str]) -> Tensor:
        """
        tokenize texts

        Args:
            texts (List[str]): texts to be tokenized

        Returns:
            torch.Tensor: tokenized texts
        """

        token_ids = []
        for text in texts:
            text = re.sub(r",", " , ", text)
            text = re.sub(r"!", " ! ", text)
            text = re.sub(r"\(", " \( ", text)
            text = re.sub(r"\)", " \) ", text)
            text = re.sub(r"\?", " \? ", text)
            text = re.sub(r"\s{2,}", " ", text).strip().lower()

            if self.language == 'zh':
                words = [
                    word for word in jieba.cut(text)
                    if word not in self.stop_words
                ]
            else:
                words = [
                    word for word in text.split()
                    if word not in self.stop_words
                ]

            token_id = [self.vocab[word] for word in words]
            real_len = len(token_id)
            if real_len < self.max_len:
                # padding zero in the front
                token_id = [0] * (self.max_len - real_len) + token_id
            elif real_len > self.max_len:
                token_id = token_id[:self.max_len]
            token_ids.append(token_id)

        return torch.tensor(token_ids)


def transform_mfan(path: str) -> torch.Tensor:
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return trans(img)


def load_adj_matrix_mfan(path: str, node_num: int):
    """
    load adjacence matrix for MFAN

    Args:
        path (str): path of adjacence list file
        node_num (int): number of nodes

    Returns:
        torch.Tensor: adjacence matrix, shape=(node_num, node_num)
    """

    with open(path, 'r') as f:
        adj_dict = json.load(f)

    adj_matrix = torch.zeros(size=(node_num, node_num))
    for ori_node, des_nodes in adj_dict.items():
        adj_matrix[int(ori_node), des_nodes] = 1
    return adj_matrix


def run_mfan(train_path: str,
             node_embedding: torch.Tensor,
             node_num: int,
             adj_matrix: torch.Tensor,
             vocab: Dict[str, int],
             word_vectors: torch.Tensor,
             max_len=50,
             batch_size=64,
             num_epochs=20,
             lr=2e-3,
             metrics: List = None,
             validate_path: str = None,
             test_path: str = None,
             device='cpu'):
    """
    run MFAN, including training, validation and testing.
    If validate_path and test_path are None, only training is performed.

    Args:
        train_path (str): path of train data
        node_embedding (torch.Tensor): node embedding, shape=(node_num, node_embedding_dim)
        node_num (int): number of nodes
        adj_matrix (torch.Tensor): adjacence matrix, shape=(node_num, node_num)
        vocab (Dict[str, int]): vocabulary dict
        word_vectors (torch.Tensor): word vectors, shape=(vocab_size, word_vector_dim)
        max_len (int): max length of text, default=50
        batch_size (int): batch size, default=64
        num_epochs (int): number of epochs, default=20
        lr (float): learning rate, default=2e-3
        metrics (List): metrics to evaluate, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        validate_path (str): path of validate data, default=None
        test_path (str): path of test data, default=None
        device (str): device to run, default='cpu'
    """

    tokenize = TokenizerMFAN(vocab, max_len)

    train_data = MultiModalDataset(train_path, ['text'], tokenize, ['image'],
                                   transform_mfan)
    train_loader = DataLoader(train_data, batch_size, True)

    if validate_path:
        validate_data = MultiModalDataset(validate_path, ['text'], tokenize,
                                          ['image'], transform_mfan)
        validate_loader = DataLoader(validate_data, batch_size, True)
    else:
        validate_loader = None

    model = MFAN(word_vectors,
                 node_num,
                 node_embedding,
                 adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    evaluator = Evaluator(metrics)
    trainer = MFANTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, num_epochs, validate_loader)

    if test_path is not None:
        test_data = MultiModalDataset(test_path, ['text'], tokenize, ['image'],
                                      transform_mfan)
        test_loader = DataLoader(test_data, batch_size, True)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def _parse_kargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    parse kargs from config dict

    Args:
        config (Dict[str, Any]): config dict, keys are the same as the args of `run_mfan`

    Returns:
        Dict[str, Any]: converted kargs
    """

    with open(config['vocab'], 'rb') as f:
        config['vocab'] = pickle.load(f)
    with open(config['word_vectors'], 'rb') as f:
        config['word_vectors'] = pickle.load(f)
    with open(config['node_embedding'], 'rb') as f:
        config['node_embedding'] = pickle.load(f)
    config['adj_matrix'] = load_adj_matrix_mfan(config['adj_matrix'],
                                                config['node_num'])

    return config


def run_mfan_from_yaml(path: str):
    """
    run MFAN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_mfan(**_parse_kargs(_config))
