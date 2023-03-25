from typing import Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from data.process.lsh import lsh_data_selection
from evaluate.evaluator import Evaluator
from model.model import AbstractModel
from train.trainer import BaseTrainer


class EDDFTrainer(BaseTrainer):
    def __init__(self, model: AbstractModel, evaluator: Evaluator, optimizer: Optimizer, budget_size=0.8,
                 num_h=10):
        super().__init__(model, evaluator, optimizer)
        self.num_h = num_h
        self.budget_size = budget_size

    def _split_train_validate(self, train_data: torch.utils.data.Dataset,
                              validate_data: Optional[torch.utils.data.Dataset] = None,
                              validate_size: Optional[float] = None):
        domain_embeddings = train_data.tensors[1]
        selected_ids = lsh_data_selection(domain_embeddings, int(len(train_data) * self.budget_size), self.num_h)
        train_tensor_data = TensorDataset(*train_data[selected_ids])
        return super()._split_train_validate(train_tensor_data, validate_data, validate_size)
