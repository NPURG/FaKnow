import torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected

from template.evaluate.evaluator import Evaluator
from template.model.social_context.base_gnn import SAGE
from template.train.gnn_trainer import GNNTrainer


def run_gnn(root: str, name: str, feature: str, batch_size: int, epochs=100, hidden_size=128):
    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
    test_dataset = UPFD(root, name, feature, 'test', ToUndirected())

    model = SAGE(num_classes=train_dataset.num_classes, num_features=train_dataset.num_features,
                 hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = GNNTrainer(model, evaluator, optimizer)
    trainer.fit(train_dataset, batch_size=batch_size, epochs=epochs, validate_data=val_dataset)
    test_result = trainer.evaluate(test_dataset, batch_size=batch_size)
    print(f'test result={test_result}')


if __name__ == '__main__':
    root = "E:\\dataset\\UPFD_Dataset"
    name = "politifact"
    feature = "profile"
    batch_size = 20
    run_gnn(root, name, feature, batch_size)