import pickle

import torch.optim
from torch.utils.data import TensorDataset

from evaluate.evaluator import Evaluator
from model.social_context.eddfn import EDDFN
from train.eddf_trainer import EDDFTrainer


def run_eddf():
    train_pool_input, train_pool_label, train_pool_domain_embedding, _ = pickle.load(open('F:\\code\\python\\cross-domain-fake-news-detection-aaai2021\\test\\train_pool.pkl', 'rb'))
    input_size = train_pool_input.shape[-1]
    domain_size = train_pool_domain_embedding.shape[-1]
    train_data = TensorDataset(torch.from_numpy(train_pool_input).float(), torch.from_numpy(train_pool_domain_embedding).float(), torch.from_numpy(train_pool_label).float())

    model = EDDFN(input_size, domain_size)
    evaluator = Evaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    trainer = EDDFTrainer(model, evaluator, optimizer)
    trainer.fit(train_data, batch_size=64, epochs=100)


if __name__ == '__main__':
    run_eddf()