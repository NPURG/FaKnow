import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from faknow.data.dataset.utils import re_split_dataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.upfd import UPFDSAGE
from faknow.train.base_gnn_trainer import BaseGNNTrainer


def run_gnn(root: str,
            name: str,
            feature: str,
            batch_size=128,
            epochs=75):
    torch.manual_seed(777)
    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
    test_dataset = UPFD(root, name, feature, 'test', ToUndirected())

    training_set, validation_set, testing_set = re_split_dataset(
        [train_dataset, test_dataset, val_dataset],
        [len(train_dataset),
         len(val_dataset),
         len(test_dataset)])

    train_loader = DataLoader(training_set,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(validation_set,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(testing_set,
                             batch_size=batch_size,
                             shuffle=False)

    feature_size = train_dataset.num_features
    model = UPFDSAGE(feature_size)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=0.01)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = BaseGNNTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, epochs, val_loader)
    test_result = trainer.evaluate(test_loader)
    print(f'test result={test_result}')


if __name__ == '__main__':
    root = "F:\\dataset\\UPFD_Dataset"
    name = "politifact"
    feature = "profile"
    run_gnn(root, name, feature)
