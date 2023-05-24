import math

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split
from transformers import get_linear_schedule_with_warmup

from faknow.data.dataset.finerfact_dataset import FinerFactDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.knowledge_aware.finerfact import FinerFact
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


def run_finerfact(train_path, test_path, bert_name):
    # data
    token_ids_train, masks_train, type_ids_train, labels_train, R_p_train, R_u_train, R_k_train, user_metadata_train, user_embeds_train = torch.load(
        train_path)
    token_ids_test, masks_test, type_ids_test, labels_test, R_p_test, R_u_test, R_k_test, user_metadata_test, user_embeds_test = torch.load(
        test_path)

    data_set = FinerFactDataset(token_ids_train, masks_train, type_ids_train,
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
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=8,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=8,
                                              shuffle=False)

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
    num_epochs, batch_size, gradient_accumulation_steps = 20, 8, 8
    t_total = int(len(train_loader) / gradient_accumulation_steps * num_epochs)
    warmup_ratio = 0.6
    warmup_steps = math.ceil(t_total * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epochs=num_epochs, validate_loader=val_loader)
    test_result = trainer.evaluate(test_loader)
    print(f"test result: {dict2str(test_result)}")


def main():
    bert_name = r'F:\code\python\FinerFact_CPU\bert_base'
    train_path = r'F:\dataset\FinerFact\Trainset_bert-base-cased_politifact_130_5.pt'
    test_path = r'F:\dataset\FinerFact\Testset_bert-base-cased_politifact_130_5.pt'
    run_finerfact(train_path, test_path, bert_name)


if __name__ == '__main__':
    main()
