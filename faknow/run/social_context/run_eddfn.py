import pickle
from typing import List

import torch
import torch.optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.eddfn import EDDFN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import lsh_data_selection


def run_eddfn(train_pool_input: Tensor,
              train_pool_label: Tensor,
              domain_embedding: Tensor,
              budget_size=0.8,
              num_h=10,
              batch_size=32,
              num_epochs=100,
              lr=0.02,
              metrics: List = None):
    input_size = train_pool_input.shape[-1]
    domain_size = domain_embedding.shape[-1]

    train_pool_set = TensorDataset(train_pool_input, domain_embedding, train_pool_label)

    selected_ids = lsh_data_selection(domain_embedding,
                                      int(len(train_pool_set) * budget_size),
                                      num_h)
    train_set = TensorDataset(*train_pool_set[selected_ids])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = EDDFN(input_size, domain_size)
    evaluator = Evaluator(metrics)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, num_epochs=num_epochs)


def main():
    train_pool_input, train_pool_label, train_pool_domain_embedding, _ = pickle.load(
        open(
            'F:\\code\\python\\cross-domain-fake-news-detection-aaai2021\\test\\train_pool.pkl',
            'rb'))
    train_pool_input = torch.from_numpy(train_pool_input).float()
    train_pool_label = torch.from_numpy(train_pool_label).float()
    domain_embedding = torch.from_numpy(train_pool_domain_embedding).float()
    run_eddfn(train_pool_input, train_pool_label, domain_embedding)


if __name__ == '__main__':
    main()
