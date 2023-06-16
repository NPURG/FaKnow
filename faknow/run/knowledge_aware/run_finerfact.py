import math
from typing import Dict, Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup

from faknow.data.dataset.finerfact_dataset import FinerFactDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.knowledge_aware.finerfact import FinerFact
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str

__all__ = ['run_finerfact', 'run_finerfact_from_yaml']


def run_finerfact(train_data,
                  bert='bert-base-uncased',
                  test_data=None,
                  val_data=None,
                  lr=5e-5,
                  batch_size=8,
                  num_epochs=20,
                  gradient_accumulation_steps=8,
                  warmup_ratio=0.6,
                  metrics=None):
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
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epochs=num_epochs, validate_loader=val_loader)

    if test_data is not None:
        test_set = FinerFactDataset(*test_data)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def _load_data(path: str):
    token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds = torch.load(
        path)
    return token_ids, masks, type_ids, labels, R_p, R_u, R_k, user_metadata, user_embeds


def run_finerfact_from_yaml(config: Dict[str, Any]):
    config['train_data'] = _load_data('train_data')
    config['test_data'] = _load_data('test_data')
    run_finerfact(**config)


if __name__ == '__main__':
    with open(r'..\..\properties\finerfact.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_finerfact_from_yaml(_config)
