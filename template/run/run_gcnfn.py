import torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected

from template.evaluate.evaluator import Evaluator
from template.model.social_context.gcnfn import GCNFN
from template.train.gnn_trainer import GNNTrainer


def run_gnn(root: str,
            name: str,
            feature: str,
            batch_size: int,
            epochs=100,
            hidden_size=128):
    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
    test_dataset = UPFD(root, name, feature, 'test', ToUndirected())

    model = GCNFN(train_dataset.num_features, hidden_size, True)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=0.01)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = GNNTrainer(model, evaluator, optimizer)
    trainer.fit(train_dataset,
                batch_size=batch_size,
                epochs=epochs,
                validate_data=val_dataset)
    test_result = trainer.evaluate(test_dataset, batch_size=batch_size)
    print(f'test result={test_result}')


if __name__ == '__main__':
    """
    Vanilla GCNFN: concat = False, feature = content
    UPFD-GCNFN: concat = True, feature = spacy
    """
    root = "E:\\dataset\\UPFD_Dataset"
    name = "politifact"
    feature = "spacy"
    batch_size = 20
    run_gnn(root, name, feature, batch_size)
