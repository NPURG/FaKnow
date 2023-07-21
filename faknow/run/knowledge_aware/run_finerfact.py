import math
from typing import Dict, Any, Tuple, Optional

import torch
import yaml
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from faknow.data.dataset.finerfact_dataset import FinerFactDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.knowledge_aware.finerfact import FinerFact
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str

__all__ = ['run_finerfact', 'run_finerfact_from_yaml']


def run_finerfact(train_data: Tuple[torch.Tensor],
                  bert='bert-base-uncased',
                  test_data: Optional[Tuple[torch.Tensor]] = None,
                  val_data: Optional[Tuple[torch.Tensor]] = None,
                  lr=5e-5,
                  batch_size=8,
                  num_epochs=20,
                  gradient_accumulation_steps=8,
                  warmup_ratio=0.6,
                  metrics=None,
                  device='cpu'):
    """
    run FinerFact, including training, validation and testing.
    If validate_path and test_path are None, only training is performed.

    Args:
        train_data (Tuple[torch.Tensor]): training data, including token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds
        bert (str): bert model, default='bert-base-uncased'
        test_data (Optional[Tuple[torch.Tensor]]): test data, including token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds, default=None
        val_data (Optional[Tuple[torch.Tensor]]): validation data, including token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds, default=None
        lr (float): learning rate, default=5e-5
        batch_size (int): batch size, default=8
        num_epochs (int): number of epochs, default=20
        gradient_accumulation_steps (int): gradient accumulation steps, default=8
        warmup_ratio (float): warmup ratio, default=0.6
        metrics (List): metrics for evaluation, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        device (str): device, default='cpu'
    """

    train_set = FinerFactDataset(*train_data)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    if val_data is not None:
        val_set = FinerFactDataset(*val_data)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    else:
        val_loader = None

    model = FinerFact(bert)

    # optimizer
    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in named_params if not any(nd in n for nd in no_decay)],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in named_params if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # scheduler
    t_total = int(len(train_loader) / gradient_accumulation_steps * num_epochs)
    warmup_steps = math.ceil(t_total * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
    trainer.fit(train_loader, num_epochs=num_epochs, validate_loader=val_loader)

    if test_data is not None:
        test_set = FinerFactDataset(*test_data)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def _load_data(path: str) -> Tuple[torch.Tensor]:
    """
    load data for FinerFact

    Args:
        path (str): path of the data file

    Returns:
        Tuple[torch.Tensor]: token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds
    """

    token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds = torch.load(
        path)
    return token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds


def _parse_kargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    parse kargs from config dict

    Args:
        config (Dict[str, Any]): config dict, keys are the same as the args of `run_finerfact`

    Returns:
        Dict[str, Any]: converted kargs
    """

    config['train_data'] = _load_data('train_data')
    config['test_data'] = _load_data('test_data')
    return config


def run_finerfact_from_yaml(path: str):
    """
    run FinerFact from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_finerfact(**_parse_kargs(_config))
