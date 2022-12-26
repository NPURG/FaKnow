import torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected, ToDense

from template.evaluate.evaluator import Evaluator
from template.model.social_context.gnncl import GNNCL
from template.train.gnn_trainer import GNNTrainer


def run_gnncl(root: str, name: str, feature: str, batch_size: int, max_nodes, epochs=60):
    train_dataset = UPFD(root, name, feature, 'train', transform=ToDense(max_nodes), pre_transform=ToUndirected())
    val_dataset = UPFD(root, name, feature, 'val', transform=ToDense(max_nodes), pre_transform=ToUndirected())
    test_dataset = UPFD(root, name, feature, 'test', transform=ToDense(max_nodes), pre_transform=ToUndirected())

    model = GNNCL(train_dataset.num_features, train_dataset.num_classes, max_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
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
    max_nodes = 500
    run_gnncl(root, name, feature, batch_size, max_nodes)