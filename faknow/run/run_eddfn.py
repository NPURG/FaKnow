import pickle

import torch.optim
from torch.utils.data import TensorDataset, DataLoader

from faknow.data.process.lsh import lsh_data_selection
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.eddfn import EDDFN
from faknow.train.trainer import BaseTrainer


def run_eddf():
    train_pool_input, train_pool_label, train_pool_domain_embedding, _ = pickle.load(
        open('F:\\code\\python\\cross-domain-fake-news-detection-aaai2021\\test\\train_pool.pkl', 'rb'))
    input_size = train_pool_input.shape[-1]
    domain_size = train_pool_domain_embedding.shape[-1]
    domain_embeddings = torch.from_numpy(train_pool_domain_embedding).float()
    train_pool_set = TensorDataset(torch.from_numpy(train_pool_input).float(), domain_embeddings,
                                   torch.from_numpy(train_pool_label).float())

    budget_size = 0.8
    num_h = 10
    selected_ids = lsh_data_selection(domain_embeddings, int(len(train_pool_set) * budget_size), num_h)
    train_set = TensorDataset(*train_pool_set[selected_ids])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    model = EDDFN(input_size, domain_size)
    evaluator = Evaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, num_epoch=100)


if __name__ == '__main__':
    run_eddf()
