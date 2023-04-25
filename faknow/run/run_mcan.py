from typing import List, Dict

import torch
from torchvision.transforms import transforms
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.mcan import MCAN
from faknow.train.trainer import BaseTrainer


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


def get_optimizer(model,
                  lr=0.0001,
                  weight_decay=0.15,
                  bert_learning_rate=None,
                  vgg_learning_rate=None,
                  dtc_conv_learning_rate=None,
                  fusion_learning_rate=None,
                  linear_learning_rate=None,
                  classifier_learning_rate=None):
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
            + list(model.linear_image.named_parameters())
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


def run_mcan():

    # data preprocess
    tokenizer = MCANTokenizer()
    transform_vgg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transform_dct = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # dataset

    # dataloader
    batch_size = 16

    epoch_num = 100
    model = MCAN('bert-base-chinese')
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(len(train_loader), epoch_num, optimizer)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epoch=50, validate_loader=val_loader)