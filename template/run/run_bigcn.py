import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from template.data.process.graph_process import DropEdge
from template.evaluate.evaluator import Evaluator
from template.model.social_context.bigcn import BiGCN
from template.train.base_gnn_trainer import BaseGNNTrainer


def run_bigcn(root: str,
              name: str,
              feature: str,
              batch_size=128,
              epochs=45,
              hidden_size=128,
              td_drop_rate=0.2,
              bu_drop_rate=0.2):
    train_dataset = UPFD(root, name, feature, 'train',
                         DropEdge(td_drop_rate, bu_drop_rate))
    val_dataset = UPFD(root, name, feature, 'val',
                       DropEdge(td_drop_rate, bu_drop_rate))
    test_dataset = UPFD(root, name, feature, 'test',
                        DropEdge(td_drop_rate, bu_drop_rate))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    feature_size = train_dataset.num_features
    model = BiGCN(feature_size, hidden_size, hidden_size)
    lr = 0.01
    bu_params_id = list(map(id, model.BURumorGCN.parameters()))
    base_params = filter(lambda p: id(p) not in bu_params_id,
                         model.parameters())
    optimizer = torch.optim.Adam([{
        'params': base_params
    }, {
        'params': model.BURumorGCN.parameters(),
        'lr': lr / 5
    }],
        lr=lr,
        weight_decay=0.001)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = BaseGNNTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, epochs, val_loader)
    test_result = trainer.evaluate(test_loader)
    print(f'test result={test_result}')


def main():
    root = "F:\\dataset\\UPFD_Dataset"
    name = "politifact"
    feature = "profile"
    run_bigcn(root, name, feature)


if __name__ == '__main__':
    main()
