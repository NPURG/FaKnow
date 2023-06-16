from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.fftpack import fft, dct
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.mcan import MCAN
from faknow.train.trainer import BaseTrainer

__all__ = ['TokenizerMCAN', 'transform_mcan', 'process_dct_mcan', 'get_optimizer_mcan', 'run_mcan',
           'run_mcan_from_yaml']


class TokenizerMCAN:
    def __init__(self, max_len=160, bert="bert-base-chinese"):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert)

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(texts,
                                return_tensors='pt',
                                max_length=self.max_len,
                                add_special_tokens=True,
                                padding='max_length',
                                truncation=True)
        return {'token_id': inputs['input_ids'], 'mask': inputs['attention_mask']}


def transform_mcan(path: str) -> Dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        img = Image.open(f)
        transform_vgg = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        vgg_feature = transform_vgg(img.convert('RGB'))

        transform_dct = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dct_feature = process_dct_mcan(transform_dct(img.convert('L')))

    return {'vgg': vgg_feature, 'dct': dct_feature}


def process_dct_mcan(img: torch.Tensor) -> torch.Tensor:
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    N = 8
    step = int(height / N)

    dct_img = np.zeros((1, N * N, step * step, 1), dtype=np.float32)
    fft_img = np.zeros((1, N * N, step * step, 1))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            block1 = block.reshape((-1, step * step, 1))  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]
            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img


def get_optimizer_mcan(model: MCAN,
                       lr=0.0001,
                       weight_decay=0.15,
                       bert_lr=1e-5,
                       vgg_lr=1e-5,
                       dtc_lr=1e-5,
                       fusion_lr=1e-2,
                       linear_lr=1e-2,
                       classifier_lr=1e-2):
    no_decay = [
        "bias",
        "gamma",
        "beta",
        "LayerNorm.weight",
        "bn_text.weight",
        "bn_dct.weight",
        "bn_1.weight",
    ]

    bert_params = list(model.bert.named_parameters())
    vgg_params = list(model.vgg.named_parameters())
    dtc_params = list(model.dct_img.named_parameters())
    fusion_params = list(
        model.fusion_layers.named_parameters()
    )
    linear_params = (
            list(model.linear_text.named_parameters())
            + list(model.linear_vgg.named_parameters())
            + list(model.linear_dct.named_parameters())
    )
    classifier_params = list(model.linear1.named_parameters()) + list(
        model.linear2.named_parameters()
    )

    optimizer_grouped_parameters = [
        # bert_param_optimizer
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": bert_lr, },
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": bert_lr, },

        # vgg_param_optimizer
        {"params": [p for n, p in vgg_params if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": vgg_lr, },
        {"params": [p for n, p in vgg_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": vgg_lr, },

        # dtc_conv_param_optimizer
        {"params": [p for n, p in dtc_params if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": dtc_lr, },
        {"params": [p for n, p in dtc_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": dtc_lr, },

        # fusion_param_optimizer
        {"params": [p for n, p in fusion_params if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": fusion_lr, },
        {"params": [p for n, p in fusion_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": fusion_lr, },

        # linear_param_optimizer
        {"params": [p for n, p in linear_params if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": linear_lr, },
        {"params": [p for n, p in linear_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": linear_lr, },

        # classifier_param_optimizer
        {"params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": classifier_lr, },
        {"params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": classifier_lr, },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def get_scheduler(batch_num, epoch_num, optimizer, warm_up_percentage=0.1):
    # Total number of training steps is number of batches * number of epochs.
    total_steps = batch_num * epoch_num

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warm_up_percentage),
        num_training_steps=total_steps
    )
    return scheduler


def run_mcan(train_path: str,
             bert='bert-base-chinese',
             max_len=255,
             batch_size=16,
             num_epochs=100,
             metrics: List = None,
             validate_path: str = None,
             test_path: str = None,
             **optimizer_kargs):
    tokenizer = TokenizerMCAN(max_len, bert)

    train_dataset = MultiModalDataset(train_path, ['text'], tokenizer, ['image'], transform_mcan)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if validate_path:
        validation_dataset = MultiModalDataset(validate_path, ['text'], tokenizer, ['image'], transform_mcan)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    model = MCAN(bert)
    optimizer = get_optimizer_mcan(model, **optimizer_kargs)
    scheduler = get_scheduler(len(train_loader), num_epochs, optimizer)
    evaluator = Evaluator(metrics)
    clip_grad_norm = {'max_norm': 1.0}

    trainer = BaseTrainer(model, evaluator, optimizer, scheduler, clip_grad_norm)
    trainer.fit(train_loader, num_epochs=num_epochs, validate_loader=val_loader)

    if test_path:
        test_dataset = MultiModalDataset(test_path, ['text'], tokenizer, ['image'], transform_mcan)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(test_result)


def run_mcan_from_yaml(config: Dict[str, Any]):
    run_mcan(**config)


if __name__ == '__main__':
    with open(r'..\..\..\properties\mcan.yaml', 'r') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_mcan_from_yaml(_config)
