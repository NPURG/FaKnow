import torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected

from template.data.process.graph_process import DropEdge
from template.evaluate.evaluator import Evaluator
from template.model.social_context.base_gnn import SAGE
from template.model.social_context.bigcn import BiGCN
from template.train.gnn_trainer import GNNTrainer


def run_bigcn(root: str, name: str, feature: str, batch_size: int, epochs=45, hidden_size=128,
              td_drop_rate=0.2, bu_drop_rate=0.2):
    train_dataset = UPFD(root, name, feature, 'train', DropEdge(td_drop_rate, bu_drop_rate))
    val_dataset = UPFD(root, name, feature, 'val', DropEdge(td_drop_rate, bu_drop_rate))
    test_dataset = UPFD(root, name, feature, 'test', DropEdge(td_drop_rate, bu_drop_rate))

    model = BiGCN(train_dataset.num_features, hidden_size, hidden_size)
    lr = 0.01
    bu_params_id = list(map(id, model.BURumorGCN.parameters()))
    base_params = filter(lambda p: id(p) not in bu_params_id, model.parameters())
    optimizer = torch.optim.Adam(
        [{
            'params': base_params
        }, {
            'params': model.BURumorGCN.parameters(),
            'lr': lr / 5
        }],
        lr=lr,
        weight_decay=0.001)
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
    run_bigcn(root, name, feature, batch_size)
