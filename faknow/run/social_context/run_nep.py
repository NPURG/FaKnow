import pickle
from typing import List, Any, Dict

import yaml
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

from faknow.data.dataset.nep_dataset import NEPDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.nep import NEP
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str

__all__ = ['run_nep', 'run_nep_from_yaml']


def run_nep(post_simcse: Tensor,
            avg_mac: Tensor,
            avg_mic: Tensor,
            p_mac: Tensor,
            p_mic: Tensor,
            avg_mic_mic: Tensor,
            token: Tensor,
            label: Tensor,
            data_ratio: List[float] = None,
            batch_size=8,
            num_epochs=10,
            lr=5e-4,
            metrics=None,
            device='cpu',
            **kwargs):
    """
    run NEP

    Args:
        post_simcse (Tensor): post simcse
        avg_mac (Tensor): avgerage macro envrionment
        avg_mic (Tensor): avgerage micro envrionment
        p_mac (Tensor): post and macro environment
        p_mic (Tensor): post and micro environment
        avg_mic_mic (Tensor): avgerage micro envrionment and micro envrionment
        token (Tensor): token ids
        label (Tensor): label
        data_ratio (List[float]): data ratio of train, val and test set, if None, [0.7, 0.2, 0.1] is used, default=None
        batch_size (int): batch size, default=8
        num_epochs (int): number of epochs, default=10
        lr (float): learning rate, default=5e-4
        metrics (List): evaluation metrics, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        device (str): device, default='cpu'
    """

    dataset = NEPDataset(post_simcse, avg_mac, avg_mic, p_mac, p_mic,
                         avg_mic_mic, token, label)

    # split dataset
    if data_ratio is None:
        data_ratio = [0.7, 0.1, 0.2]

    train_size = int(data_ratio[0] * len(dataset))
    val_size = int(data_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = NEP()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr)
    evaluator = Evaluator(metrics)

    trainer = BaseTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, num_epochs, val_loader)

    test_result = trainer.evaluate(test_loader)
    print(f"test result: {dict2str(test_result)}")


def _parse_kargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    parse kargs from config dict

    Args:
        config (Dict[str, Any]): config dict, keys are the same as the args of `run_nep`

    Returns:
        Dict[str, Any]: converted kargs
    """

    with open(config.pop('data'), 'rb') as f:
        config.update(pickle.load(f))
    return config


def run_nep_from_yaml(path: str):
    """
    run NEP from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_nep(**_parse_kargs(_config))
