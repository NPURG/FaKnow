import torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected

from template.data.dataset.utils import re_split_dataset
from template.evaluate.evaluator import Evaluator
from template.model.social_context.base_gnn import SAGE
from template.train.gnn_trainer import GNNTrainer


def run_gnn(root: str,
            name: str,
            feature: str,
            batch_size=128,
            epochs=75,
            hidden_size=128):
    torch.manual_seed(777)
    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
    test_dataset = UPFD(root, name, feature, 'test', ToUndirected())

    training_set, validation_set, testing_set = re_split_dataset(
        [train_dataset, test_dataset, val_dataset],
        [len(train_dataset),
         len(val_dataset),
         len(test_dataset)])

    feature_size = train_dataset.num_features
    model = SAGE(feature_size, hidden_size, True)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=0.01)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = GNNTrainer(model, evaluator, optimizer)
    trainer.fit(training_set,
                batch_size=batch_size,
                epochs=epochs,
                validate_data=validation_set)
    test_result = trainer.evaluate(testing_set, batch_size=batch_size)
    print(f'test result={test_result}')


if __name__ == '__main__':
    root = "E:\\dataset\\UPFD_Dataset"
    name = "politifact"
    feature = "profile"
    run_gnn(root, name, feature)
