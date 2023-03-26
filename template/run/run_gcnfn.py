import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from template.evaluate.evaluator import Evaluator
from template.model.social_context.gcnfn import GCNFN
from template.train.base_gnn_trainer import BaseGNNTrainer


def run_gcnfn(root: str,
              name: str,
              feature: str,
              batch_size=128,
              epochs=110,
              hidden_size=128):
    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
    test_dataset = UPFD(root, name, feature, 'test', ToUndirected())

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    model = GCNFN(train_dataset.num_features, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=0.01)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])
    trainer = BaseGNNTrainer(model, evaluator, optimizer)

    trainer.fit(train_loader, epochs, val_loader)
    test_result = trainer.evaluate(test_loader)
    print(f'test result={test_result}')


def main():
    """
        Vanilla _BaseGCNFN: concat = False, feature = content
        UPFD-_BaseGCNFN: concat = True, feature = spacy
        """
    root = "F:\\dataset\\UPFD_Dataset"
    name = "politifact"
    feature = "content"
    run_gcnfn(root, name, feature)

if __name__ == '__main__':
    main()
