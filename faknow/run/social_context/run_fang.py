from faknow.data.dataset.fang_dataset import FangDataset, FangTrainDataSet
from faknow.model.social_context.fang import FANG
from faknow.evaluate.evaluator import Evaluator
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str

import torch
import yaml

__all__ = ['run_fang', 'run_fang_from_yaml']

def run_fang(data_root: str,
             metrics=None,
             lr=1e-4,
             weight_decay=5e-4,
             batch_size=32,
             num_epochs=100,
             input_size=100,
             embedding_size=16,
             num_stance=4,
             num_stance_hidden=4,
             timestamp_size=2,
             num_classes=2,
             dropout=0.1,
             device='cpu'
             ):
    """
     run FANG, including training, validation and testing.
    If validate_path and test_path are None, only training is performed

    Args:
        data_root(str): the data path. including entities.txt, entity_features.tsv, source_citation.tsv, source_publication.tsv, user_relationships.tsv, news_info.tsv, report.tsv, support_neutral.tsv, support_negative.tsv, deny.tsv. example referring to dataset/example/FANG.
        metrics (List): metrics for evaluation, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        lr(float): learning rate. default=1e-4.
        weight_decay(float): weight dacay. default=5e-4.
        batch_size(int): batch size. default=32.
        num_epochs(int): epoch num. default=100.
        input_size(int): embedding size of raw feature. default=100.
        embedding_size(int): graphsage output size. default=16.
        num_stance(int): total num of stance. default=4.
        num_stance_hidden(int): stance's embedding size. please let num_stance_hidden * num_stance = embedding_size. default=4.
        timestamp_size(int): timestamp's embedding size. default=2
        num_classes(int): label num. default=2.
        dropout(float): dropout rate. default=0.1.
        device(str): compute device. default='cpu'.
    """
    fang_data = FangDataset(data_root)
    train_idxs = fang_data.train_idxs
    train_loader = torch.utils.data.DataLoader(train_idxs,
                                              batch_size=batch_size,
                                              shuffle=True
                                              )
    if fang_data.dev_idxs is not None:
        val_data = fang_data.dev_idxs
        val_label = [fang_data.news_labels[val_node] for val_node in val_data]
        val_indexs =  FangTrainDataSet(val_data, val_label)
        val_loader = torch.utils.data.DataLoader(val_indexs,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    model = FANG(fang_data, input_size, embedding_size, num_stance, num_stance_hidden,
                 timestamp_size,num_classes,dropout)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model, evaluator, optimizer,device=device)

    trainer.fit(train_loader, num_epochs=num_epochs, validate_loader=val_loader)

    if fang_data.test_idxs is not None:
        test_data = fang_data.test_idxs
        test_label = [fang_data.news_labels[test_node] for test_node in test_data]
        test_indexs = FangTrainDataSet(test_data, test_label)
        test_loader = torch.utils.data.DataLoader(test_indexs,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")

def run_fang_from_yaml(path: str):
    """
        run FANG from yaml config file

        Args:
            path (str): yaml config file path
        """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_fang(**_config)