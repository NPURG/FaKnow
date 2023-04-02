import math

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split
from transformers import get_linear_schedule_with_warmup

from evaluate.evaluator import Evaluator
from model.knowledge_aware.finerfact import FinerFact
from train.trainer import BaseTrainer
from utils.util import dict2str


def collate_fn(batch):
    """stack data in batch"""
    token_id = torch.stack([item[0] for item in batch])
    mask = torch.stack([item[1] for item in batch])
    type_id = torch.stack([item[2] for item in batch])
    label = torch.stack([item[3] for item in batch])
    R_p = torch.stack([item[4] for item in batch])
    R_u = torch.stack([item[5] for item in batch])
    R_k = torch.stack([item[6] for item in batch])
    user_metadata = torch.stack([item[7] for item in batch])

    return {
        'token_id': token_id,
        'mask': mask,
        'type_id': type_id,
        'label': label,
        'R_p': R_p,
        'R_u': R_u,
        'R_k': R_k,
        'user_metadata': user_metadata
    }


def run_finerfact(train_path, test_path, bert_name):
    # data
    token_ids_train, masks_train, type_ids_train, labels_train, R_p_train, R_u_train, R_k_train, user_metadata_train, user_embeds_train = torch.load(
        train_path)
    token_ids_test, masks_test, type_ids_test, labels_test, R_p_test, R_u_test, R_k_test, user_metadata_test, user_embeds_test = torch.load(
        test_path)

    data_set = TensorDataset(token_ids_train, masks_train, type_ids_train,
                             labels_train, R_p_train, R_u_train, R_k_train,
                             user_metadata_train)
    train_set, val_set = random_split(
        data_set,
        [int(len(data_set) * 0.8),
         len(data_set) - int(len(data_set) * 0.8)])
    test_set = TensorDataset(token_ids_test, masks_test, type_ids_test,
                             labels_test, R_p_test, R_u_test, R_k_test,
                             user_metadata_test)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=8,
                                               shuffle=True,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=8,
                                             shuffle=False,
                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=8,
                                              shuffle=False,
                                              collate_fn=collate_fn)

    model = FinerFact(bert_name)

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

    # scheduler
    num_epoch, batch_size, gradient_accumulation_steps = 20, 8, 8
    t_total = int(len(train_loader) / gradient_accumulation_steps * num_epoch)
    warmup_ratio = 0.6
    warmup_steps = math.ceil(t_total * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epoch=num_epoch, validate_loader=val_loader)
    test_result = trainer.evaluate(test_loader)
    print(f"test result: {dict2str(test_result)}")


def main():
    bert_name = r'F:\code\python\FinerFact_CPU\bert_base'
    train_path = r'F:\dataset\FinerFact\Trainset_bert-base-cased_politifact_130_5.pt'
    test_path = r'F:\dataset\FinerFact\Testset_bert-base-cased_politifact_130_5.pt'
    run_finerfact(train_path, test_path, bert_name)


if __name__ == '__main__':
    main()
