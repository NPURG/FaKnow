import pickle
from typing import List, Any, Dict

import yaml
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

from faknow.data.dataset.nep_dataset import NEPDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.nep import NEP
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


def run_nep(post_simcse, avg_mac, avg_mic, p_mac, p_mic, avg_mic_mic, token, label,
            data_ratio: List[float] = None, batch_size=8, num_epochs=10, lr=5e-4, metrics=None, **kwargs):
    dataset = NEPDataset(post_simcse, avg_mac, avg_mic, p_mac, p_mic, avg_mic_mic, token, label)

    # split dataset
    if data_ratio is None:
        data_ratio = [0.7, 0.1, 0.2]

    train_size = int(data_ratio[0] * len(dataset))
    val_size = int(data_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = NEP(**kwargs)
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr)
    evaluator = Evaluator(metrics)

    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, num_epochs, val_loader)

    test_result = trainer.evaluate(test_loader)
    print(f"test result: {dict2str(test_result)}")


def run_nep_from_yaml(config: Dict[str, Any]):
    with open(config['data'], 'rb') as f:
        config.update(pickle.load(f))
        del config['data']
    run_nep(**config)


if __name__ == '__main__':
    with open(r'..\..\properties\nep.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_nep_from_yaml(_config)
