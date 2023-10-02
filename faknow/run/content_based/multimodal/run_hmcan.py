from typing import List

import torch
import yaml
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.data.process.text_process import TokenizerForBert
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.hmcan import HMCAN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str

__all__ = ['transform_hmcan', 'run_hmcan', 'run_hmcan_from_yaml']


def transform_hmcan(path: str) -> torch.Tensor:
    """

    transform image to tensor for HMCAN

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


def run_hmcan(train_path: str,
              max_len=20,
              left_num_layers=2,
              left_num_heads=12,
              dropout=0.1,
              right_num_layers=2,
              right_num_heads=12,
              alpha=0.7,
              batch_size=256,
              lr=0.001,
              num_epochs=150,
              bert='bert-base-uncased',
              metrics: List = None,
              validate_path: str = None,
              test_path: str = None,
              device='cpu') -> None:
    """
    run HMCAN, including training, validation and testing.
    If validate_path and test_path are None, only training is performed.

    Args:
        train_path (str): path of the training set
        max_len (int): max length of the text, default=20
        left_num_layers(int): the numbers of  the left Attention&FFN layer
            in Contextual Transformer, Default=2.
        left_num_heads(int): the numbers of head in
            Multi-Head Attention layer(in the left Attention&FFN), Default=12.
        dropout(float): dropout rate, Default=0.1.
        right_num_layers(int): the numbers of  the right Attention&FFN layer
            in Contextual Transformer, Default=2.
        right_num_heads(int): the numbers of head in
            Multi-Head Attention layer(in the right Attention&FFN), Default=12.
        alpha(float): the weight of the first Attention&FFN layer's output,
            Default=0.7.
        batch_size (int): batch size, default=256
        lr (float): learning rate, default=0.001
        num_epochs (int): number of epochs, default=150
        bert (str): bert model name, default='bert-base-uncased'
        metrics (List): metrics, if None,
            ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        validate_path (str): path of the validation set, default=None
        test_path (str): path of the test set, default=None
        device (str): device, default='cpu'
    """

    tokenizer = TokenizerForBert(max_len, bert)
    train_set = MultiModalDataset(train_path, ['text'], tokenizer, ['image'],
                                  transform_hmcan)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    if validate_path is not None:
        val_set = MultiModalDataset(validate_path, ['text'], tokenizer,
                                    ['image'], transform_hmcan)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
    else:
        val_loader = None

    model = HMCAN(max_len, left_num_layers, left_num_heads, dropout,
                  right_num_layers, right_num_heads, alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = MultiModalDataset(test_path, ['text'], tokenizer, ['image'],
                                     transform_hmcan)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def run_hmcan_from_yaml(path: str):
    """
    run HMCAN yaml config file

    Args:
        path(str): yaml config file path

    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_hmcan(**_config)
