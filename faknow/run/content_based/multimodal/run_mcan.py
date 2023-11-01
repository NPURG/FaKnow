from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.fftpack import fft, dct
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from transformers import get_linear_schedule_with_warmup

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.data.process.text_process import TokenizerForBert
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.mcan import MCAN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str, EarlyStopping

__all__ = [
    'transform_mcan', 'process_dct_mcan',
    'get_optimizer_mcan', 'run_mcan', 'run_mcan_from_yaml'
]


def transform_mcan(path: str) -> Dict[str, torch.Tensor]:
    """
    transform image to tensor for MCAN

    Args:
        path (str): path of the image

    Returns:
        Dict[str, torch.Tensor]: transformed image with key 'vgg' and 'dct'
    """

    with open(path, "rb") as f:
        img = Image.open(f)
        transform_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        vgg_feature = transform_img(img.convert('RGB'))
        dct_feature = process_dct_mcan(transform_img(img.convert('L')))

    return {'vgg': vgg_feature, 'dct': dct_feature}


def process_dct_mcan(img: torch.Tensor) -> torch.Tensor:
    """
    process image with dct(Discrete Cosine Transform) for MCAN

    Args:
        img (torch.Tensor): image tensor to be processed

    Returns:
        torch.Tensor: dct processed image tensor
    """

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
            block = np.array(img[:, row:(row + step), col:(col + step)],
                             dtype=np.float32)
            block1 = block.reshape((-1, step * step, 1))  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]
            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(
        dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

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
    """
    generate optimizer for MCAN

    Args:
        model (MCAN): MCAN model
        lr (float): learning rate, default=0.0001
        weight_decay (float): weight decay, default=0.15
        bert_lr (float): learning rate of bert, default=1e-5
        vgg_lr (float): learning rate of vgg, default=1e-5
        dtc_lr (float): learning rate of dct, default=1e-5
        fusion_lr (float): learning rate of fusion layers, default=1e-2
        linear_lr (float): learning rate of linear layers, default=1e-2
        classifier_lr (float): learning rate of classifier layers, default=1e-2

    Returns:
        torch.optim.Optimizer: optimizer for MCAN
    """

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
    fusion_params = list(model.fusion_layers.named_parameters())
    linear_params = (list(model.linear_text.named_parameters()) +
                     list(model.linear_vgg.named_parameters()) +
                     list(model.linear_dct.named_parameters()))
    classifier_params = list(model.linear1.named_parameters()) + list(
        model.linear2.named_parameters())

    optimizer_grouped_parameters = [
        # bert_param_optimizer
        {
            "params":
            [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
            "weight_decay":
            weight_decay,
            "lr":
            bert_lr,
        },
        {
            "params":
            [p for n, p in bert_params if any(nd in n for nd in no_decay)],
            "weight_decay":
            0.0,
            "lr":
            bert_lr,
        },

        # vgg_param_optimizer
        {
            "params":
            [p for n, p in vgg_params if not any(nd in n for nd in no_decay)],
            "weight_decay":
            weight_decay,
            "lr":
            vgg_lr,
        },
        {
            "params":
            [p for n, p in vgg_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": vgg_lr,
        },

        # dtc_conv_param_optimizer
        {
            "params":
            [p for n, p in dtc_params if not any(nd in n for nd in no_decay)],
            "weight_decay":
            weight_decay,
            "lr":
            dtc_lr,
        },
        {
            "params":
            [p for n, p in dtc_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": dtc_lr,
        },

        # fusion_param_optimizer
        {
            "params": [
                p for n, p in fusion_params
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            weight_decay,
            "lr":
            fusion_lr,
        },
        {
            "params":
            [p for n, p in fusion_params if any(nd in n for nd in no_decay)],
            "weight_decay":
            0.0,
            "lr":
            fusion_lr,
        },

        # linear_param_optimizer
        {
            "params": [
                p for n, p in linear_params
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            weight_decay,
            "lr":
            linear_lr,
        },
        {
            "params":
            [p for n, p in linear_params if any(nd in n for nd in no_decay)],
            "weight_decay":
            0.0,
            "lr":
            linear_lr,
        },

        # classifier_param_optimizer
        {
            "params": [
                p for n, p in classifier_params
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            weight_decay,
            "lr":
            classifier_lr,
        },
        {
            "params": [
                p for n, p in classifier_params
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
            "lr":
            classifier_lr,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def get_scheduler(batch_num: int,
                  epoch_num: int,
                  optimizer: torch.optim.Optimizer,
                  warm_up_percentage=0.1):
    """
    generate scheduler for MCAN

    Args:
        batch_num (int): number of batches
        epoch_num (int): number of epochs
        optimizer (torch.optim.Optimizer): optimizer for MCAN
        warm_up_percentage (float): percentage of warm up, default=0.1

    Returns:
        torch.optim.lr_scheduler.LambdaLR: scheduler for MCAN
    """

    # Total number of training steps is number of batches * number of epochs.
    total_steps = batch_num * epoch_num

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warm_up_percentage),
        num_training_steps=total_steps)
    return scheduler


def run_mcan(train_path: str,
             bert='bert-base-chinese',
             max_len=255,
             batch_size=16,
             num_epochs=100,
             metrics: List = None,
             validate_path: str = None,
             test_path: str = None,
             patience=10,
             device='cpu',
             **optimizer_kargs):
    """
    run MCAN

    Args:
        train_path (str): path of training data
        bert (str): bert model, default='bert-base-chinese'
        max_len (int): max length of text, default=255
        batch_size (int): batch size, default=16
        num_epochs (int): number of epochs, default=100
        metrics (List): metrics,
            if None, ['accuracy', 'precision', 'recall', 'f1'] will be used,
            default=None
        validate_path (str): path of validation data, default=None
        test_path (str): path of test data, default=None
        patience (int): patience of early stopping, default=10
        device (str): device, default='cpu'
        **optimizer_kargs: optimizer kargs
    """

    tokenizer = TokenizerForBert(max_len, bert)

    train_dataset = MultiModalDataset(train_path, ['text'], tokenizer,
                                      ['image'], transform_mcan)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    if validate_path:
        validation_dataset = MultiModalDataset(validate_path, ['text'],
                                               tokenizer, ['image'],
                                               transform_mcan)
        val_loader = DataLoader(validation_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    else:
        val_loader = None

    model = MCAN(bert)
    optimizer = get_optimizer_mcan(model, **optimizer_kargs)
    scheduler = get_scheduler(len(train_loader), num_epochs, optimizer)
    evaluator = Evaluator(metrics)
    clip_grad_norm = {'max_norm': 1.0}
    early_stopping = EarlyStopping(patience)

    trainer = BaseTrainer(model,
                          evaluator,
                          optimizer,
                          scheduler,
                          clip_grad_norm,
                          device=device,
                          early_stopping=early_stopping)
    trainer.fit(train_loader,
                num_epochs=num_epochs,
                validate_loader=val_loader)

    if test_path:
        test_dataset = MultiModalDataset(test_path, ['text'], tokenizer,
                                         ['image'], transform_mcan)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_mcan_from_yaml(path: str):
    """
    run MCAN from yaml file

    Args:
        path (str): path of yaml file
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_mcan(**_config)
