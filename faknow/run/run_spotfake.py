import random

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from faknow.data.dataset.spotfake_dataset import FakeNewsDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.spotfake import SpotFake
from faknow.train.trainer_gpu import BaseTrainer

def run_spotfake(
        root: str,
        pre_trained_bert_name="bert-base-uncased",
        batch_size=8,
        epochs=50,
        MAX_LEN=500,
):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    df_train = pd.read_csv("/root/Template/dataset/example/dataset_example_SpotFake/twitter/train_posts_clean.csv")
    df_test = pd.read_csv("/root/Template/dataset/example/dataset_example_SpotFake/twitter/test_posts.csv")

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tokenizer = BertTokenizer.from_pretrained(pre_trained_bert_name, do_lower_case=True )

    training_set = FakeNewsDataset(df_train, root + "images_train/", image_transform, tokenizer, MAX_LEN)
    validation_set = FakeNewsDataset(df_test, root + "images_test/", image_transform, tokenizer, MAX_LEN)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    model = SpotFake(pre_trained_bert_name=pre_trained_bert_name)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-5,
        eps=1e-8
    )

    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = BaseTrainer(model, evaluator, optimizer)
    trainer.fit(train_loader, epochs, validation_loader)


def main():
    root = "/root/Template/dataset/example/dataset_example_SpotFake/twitter/"
    pre_trained_bert_name = "bert-base-uncased"
    run_spotfake(root, pre_trained_bert_name)


if __name__ == '__main__':
    main()