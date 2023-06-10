import pickle
import re
from typing import Dict, List

import jieba
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.eann import EANN
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str, read_stop_words


class EANNTokenizer:
    def __init__(self, vocab: Dict[str, int], max_len: int, stop_words: List[str], language='zh') -> None:
        assert language in ['zh', 'en'], "language must be one of {zh, en}"
        self.language = language
        self.vocab = vocab
        self.max_len = max_len
        self.stop_words = stop_words

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        token_ids = []
        masks = []
        for text in texts:
            cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", text).strip().lower()

            if self.language == 'zh':
                split_words = jieba.cut_for_search(cleaned_text)
                words = " ".join([word for word in split_words if word not in self.stop_words])
            else:
                words = [word for word in cleaned_text.split() if word not in self.stop_words]

            token_id = [self.vocab[word] for word in words]
            real_len = len(token_id)
            if real_len < self.max_len:
                token_id.extend([0] * (self.max_len - real_len))
            elif real_len > self.max_len:
                token_id = token_id[:self.max_len]
            token_ids.append(token_id)

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


def run_eann(train_path: str,
             vocab: Dict[str, int],
             stop_words: List[str],
             word_vectors: torch.Tensor,
             language='zh',
             max_len=255,
             batch_size=100,
             event_num: int = None,
             lr=0.001,
             num_epochs=100,
             metrics: List = None,
             validate_path: str = None,
             test_path: str = None) -> None:
    """
    todo run函数是否不要加过多参数，直接最基础最原始的几个参数就可以了，不需要很复杂，对于复杂功能就让用户自己写代码
    直接传入数据
    """

    tokenizer = EANNTokenizer(vocab, max_len, stop_words, language)

    train_set = MultiModalDataset(train_path, ['text'], tokenizer, ['image'], transform)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    if event_num is None:
        event_num = torch.max(train_set.data['domain']).item() + 1

    if validate_path is not None:
        val_set = MultiModalDataset(validate_path, ['text'], tokenizer, ['image'], transform)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
    else:
        val_loader = None

    model = EANN(event_num, embed_weight=word_vectors)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=adjust_lr)
    evaluator = Evaluator(metrics)
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

    if test_path is not None:
        test_set = MultiModalDataset(test_path, ['text'], tokenizer, ['image'], transform)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def main():
    train_path = "F:\\dataset\\dataset_example_EANN\\all\\train.json"
    test_path = "F:\\dataset\\dataset_example_EANN\\all\\test.json"
    val_path = "F:\\dataset\\dataset_example_EANN\\all\\val.json"
    word_vector_path = "F:\\code\\python\EANN-KDD18-degugged11.2\\Data\\weibo\\word_embedding.pickle"
    with open(word_vector_path, 'rb') as f:
        w2v, _, vocab, _, max_len = pickle.load(f)
    stop_words = read_stop_words("/faknow/data/process/stop_words/stop_words.txt")
    run_eann(train_path, vocab, stop_words, torch.from_numpy(w2v), max_len=max_len, validate_path=val_path,
             test_path=test_path)


if __name__ == '__main__':
    main()
