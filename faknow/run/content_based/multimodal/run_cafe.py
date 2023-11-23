from typing import List
import torch
from faknow.utils.util import dict2str
from torch.utils.data import DataLoader
from faknow.train.cafe_trainer import CafeTrainer
from faknow.evaluate.evaluator import Evaluator
from faknow.data.dataset.cafe_dataset import CafeDataset
from faknow.model.content_based.multi_modal.cafe import SimilarityModule, DetectionModule

def run_cafe(dataset_dir: str,
             num_workers=1,
             batch_size=64,
             lr=1e-3,
             l2=0,  # 1e-5
             NUM_EPOCH=100,
             metrics: List = None,
             DEVICE="cuda:0"):
    """
    run CAFE
    Args:
        dataset_dir (str): path of data,including training data and testing data.
        num_workers (int): number of epochs, default=100
        batch_size (int): batch size, default=64
        lr (float): learning rate, default=0.001
        l2 (float): weight_decay,default=0
        NUM_EPOCH(int): number of epochs, default=50
        metrics (List): evaluation metrics,
            if None, ['accuracy', 'precision', 'recall', 'f1'] is used,
            default=None
        DEVICE (str): device to run model, default='cpu'
    """
    device = torch.device(DEVICE)
    # ---  Load Data  ---
    train_set = CafeDataset(
        "{}/train_text_with_label.npz".format(dataset_dir),
        "{}/train_image_with_label.npz".format(dataset_dir)
    )
    test_set = CafeDataset(
        "{}/test_text_with_label.npz".format(dataset_dir),
        "{}/test_image_with_label.npz".format(dataset_dir)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # ---  Build Model & Trainer  ---
    similarity_module = SimilarityModule()
    similarity_module.to(device)
    detection_module = DetectionModule()
    detection_module.to(device)
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task1
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task2
    similarity_evaluator = Evaluator(metrics)
    detection_evaluator = Evaluator(metrics)

    trainer = CafeTrainer(similarity_module,
                          detection_module,
                          similarity_evaluator,
                          detection_evaluator,
                          optim_task_similarity,
                          optim_task_detection,
                          device=device)

    trainer.fit(train_loader, NUM_EPOCH)

    if test_loader is not None:
        test_result = trainer.evaluate(test_loader)
        print('test result: ', dict2str(test_result))