import pickle
import re
from typing import Dict, List

import jieba
import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.eann import EANN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


class EANNTokenizer:
    def __init__(self, vocab: Dict[str, int], max_len: int, stop_words: List[str]) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.stop_words = stop_words

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        token_ids = []
        masks = []
        for text in texts:

            # split words
            cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", text).strip().lower()
            split_words = jieba.cut_for_search(cleaned_text)
            words = " ".join([word for word in split_words if word not in self.stop_words])

            # convert to id
            token_id = [self.vocab[word] for word in words]
            real_len = len(token_id)
            if real_len < self.max_len:
                token_id.extend([0] * (self.max_len - real_len))
            elif real_len > self.max_len:
                token_id = token_id[:self.max_len]
            token_ids.append(token_id)

            # generate mask
            mask = torch.zeros(self.max_len)
            mask[:real_len] = 1
            masks.append(mask)

        return {'token_id': torch.tensor(token_ids), 'mask': torch.stack(masks)}


def transform(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(img)


def adjust_lr(epoch: int) -> float:
    return 0.001 / (1. + 10 * (float(epoch) / 100)) ** 0.75


def run_eann(path: str,
             word_vectors: torch.Tensor,
             word_idx_map: Dict[str, int],
             max_len: int,
             stop_words: List[str]):

    tokenizer = EANNTokenizer(word_idx_map, max_len, stop_words)

    dataset = MultiModalDataset(path, ['text'], tokenizer, ['image'], transform)
    event_num = torch.max(dataset.data['domain']).item() + 1

    val_size = int(len(dataset) * 0.1)
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = EANN(event_num, embed_weight=word_vectors)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=0.001)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=adjust_lr)
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epoch=100, validate_loader=val_loader)
    test_result = trainer.evaluate(test_loader)
    print(f"test result: {dict2str(test_result)}")


def main():
    path = "F:\\dataset\\dataset_example_EANN\\all\\eann.json"
    word_vector_path = "F:\\code\\python\EANN-KDD18-degugged11.2\\Data\\weibo\\word_embedding.pickle"
    with open(word_vector_path, 'rb') as f:
        weight = pickle.load(f)  # W, W2, word_idx_map, vocab
        word_vectors, _, word_idx_map, _, max_len = weight

    with open("F:\\code\\python\\FaKnow\\faknow\\data\\process\\stop_words\\stop_words.txt", 'r',
              encoding='utf-8') as f:
        stop_words = [str(line).strip() for line in f.readlines()]
    run_eann(path, torch.from_numpy(word_vectors), word_idx_map, max_len, stop_words)


if __name__ == '__main__':
    main()
