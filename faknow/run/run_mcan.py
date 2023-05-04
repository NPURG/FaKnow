import os.path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.fftpack import fft, dct
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.mcan import MCAN
from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.data.dataset.mcan_dataset import MultiModalDataset
from faknow.train.mcan_trainer import MCANTrainer


class MCANTokenizer:
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


def transform(path: str) -> Dict[str, torch.Tensor]:
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
        dct_feature = process_dct_img(transform_dct(img.convert('L')))

    return {'vgg': vgg_feature, 'dct': dct_feature}


def process_dct_img(img: torch.Tensor) -> torch.Tensor:
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


def get_optimizer(model: MCAN,
                  lr=0.0001,
                  weight_decay=0.15,
                  bert_learning_rate=1e-5,
                  vgg_learning_rate=1e-5,
                  dtc_conv_learning_rate=1e-5,
                  fusion_learning_rate=1e-2,
                  linear_learning_rate=1e-2,
                  classifier_learning_rate=1e-2):
    no_decay = [
        "bias",
        "gamma",
        "beta",
        "LayerNorm.weight",
        "bn_text.weight",
        "bn_dct.weight",
        "bn_1.weight",
    ]

    bert_param_optimizer = list(model.bert.named_parameters())
    vgg_param_optimizer = list(model.vgg.named_parameters())
    dtc_conv_param_optimizer = list(model.dct_img.named_parameters())
    fusion_param_optimizer = list(
        model.fusion_layers.named_parameters()
    )
    linear_param_optimizer = (
            list(model.linear_text.named_parameters())
            + list(model.linear_vgg.named_parameters())
            + list(model.linear_dct.named_parameters())
    )
    classifier_param_optimizer = list(model.linear1.named_parameters()) + list(
        model.linear2.named_parameters()
    )

    optimizer_grouped_parameters = [
        # bert_param_optimizer
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": bert_learning_rate, },
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": bert_learning_rate, },
        # vgg_param_optimizer
        {"params": [p for n, p in vgg_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": vgg_learning_rate, },
        {"params": [p for n, p in vgg_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": vgg_learning_rate, },
        # dtc_conv_param_optimizer
        {"params": [p for n, p in dtc_conv_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": dtc_conv_learning_rate, },
        {"params": [p for n, p in dtc_conv_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": dtc_conv_learning_rate, },
        # fusion_param_optimizer
        {"params": [p for n, p in fusion_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": fusion_learning_rate, },
        {"params": [p for n, p in fusion_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": fusion_learning_rate, },
        # linear_param_optimizer
        {"params": [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": linear_learning_rate, },
        {"params": [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": linear_learning_rate, },
        # classifier_param_optimizer
        {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay,
         "lr": classifier_learning_rate, },
        {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": classifier_learning_rate, },
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


def run_mcan(root):
    # data preprocess
    tokenizer = MCANTokenizer()

    # dataset
    train_dataset = MultiModalDataset(os.path.join(root, 'train.json'), ['text'], tokenizer, ['image'], transform)
    test_dataset = MultiModalDataset(os.path.join(root, 'test.json'), ['text'], tokenizer, ['image'], transform)
    validation_size = int(len(train_dataset) * 0.1)
    train_dataset, validation_dataset = random_split(train_dataset,
                                                     [len(train_dataset) - validation_size, validation_size])

    # dataloader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epoch_num = 100
    model = MCAN('bert-base-chinese')
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(len(train_loader), epoch_num, optimizer)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = MCANTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epoch=50, validate_loader=val_loader)
    test_result = trainer.evaluate(test_loader)
    print(test_result)


def main():
    root = r'F:\dataset\dataset_example_MCAN'
    run_mcan(root)


if __name__ == '__main__':
    main()
