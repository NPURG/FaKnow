import os
from typing import Dict, List, Any

import numpy as np
import torch
from torchvision import transforms

from data.process.text_process import build_word2vec, padding_vec_and_idx
from evaluate.evaluator import Evaluator
from model.multi_modal.eann import EANN
from template.data.dataset.multi_modal_dataset import FolderMultiModalDataset
from template.data.process.text_process import chinese_tokenize
from train.eann_trainer import EANNTrainer


def generate_event_label(event_id: int, event_label_map: Dict[int,
                                                              int]) -> int:
    if event_id not in event_label_map:
        event_label_map[event_id] = len(event_label_map)
    event_label = event_label_map[event_id]
    return event_label


def generate_max_text_len_and_event_label(path: str):
    event_labels = []
    event_label_map = {}
    max_text_len = 0

    for dir in os.listdir(path):
        for entry in os.scandir(os.path.join(path, dir)):
            # 找到路径下的文本文件
            if os.path.splitext(entry.name)[1] == ".txt":
                file_path = os.path.join(path, dir, entry.name)
                with open(file_path, encoding='utf-8') as f:
                    for i, line in enumerate(f.readlines()):
                        line = line.rstrip()

                        # 第1行 event_id
                        if (i + 1) % 2 == 1:
                            event_id = int(line)
                            event_labels.append(
                                generate_event_label(event_id,
                                                     event_label_map))

                        # 第2行 文本内容
                        if (i + 1) % 2 == 0:
                            tokens = chinese_tokenize(line)
                            if len(tokens) > max_text_len:
                                max_text_len = len(tokens)
    return max_text_len, event_labels


def word_to_idx(text: List[str], word_idx_map: Dict[str, int],
                max_text_len: int):
    """convert words in text to id"""
    # todo 优化循环
    words_id = [word_idx_map[word] for word in text]  # 把每个word转为对应id
    while len(words_id) < max_text_len:  # 填充0
        words_id.append(0)
    return words_id


def eann_embedding(path: str, other_params: Dict[str, Any]):
    with open(path, encoding='utf-8') as f:
        _ = f.readline()
        line = f.readline()  # 第二行才是文本
        tokens = chinese_tokenize(line)
    word_idx_map = other_params['word_idx_map']
    max_text_len = other_params['max_text_len']
    words_id = word_to_idx(tokens, word_idx_map, max_text_len)
    mask = torch.zeros(max_text_len, dtype=torch.int)
    mask[:len(tokens)] = 1
    return torch.tensor(words_id), mask


def run_eann(root: str,
             pre_trained_word2vec=True,
             max_text_len: int = None,
             word_vectors: np.ndarray = None,
             word_idx_map: Dict[str, int] = None):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not pre_trained_word2vec:
        word_vectors, word_idx_map, max_text_len = build_word2vec(root)
        word_vectors, word_idx_map = padding_vec_and_idx(
            word_vectors, word_idx_map)
        print(f'using corpus from {root} to train word2vec')

    embedding_params = {
        'word_idx_map': word_idx_map,
        'max_text_len': max_text_len
    }

    max_text_len, event_labels = generate_max_text_len_and_event_label(root)
    event_num = max(event_labels) + 1

    dataset = FolderMultiModalDataset(root,
                                      embedding=eann_embedding,
                                      transform=image_transforms,
                                      embedding_params=embedding_params,
                                      event_label=torch.tensor(event_labels))

    model = EANN(event_num,
                 hidden_size=32,
                 dropout=1,
                 reverse_lambd=1,
                 vocab_size=len(word_idx_map),
                 embed_weight=word_vectors)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=0.00025)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = EANNTrainer(model, evaluator, optimizer)
    trainer.fit(dataset,
                batch_size=20,
                epochs=100,
                validate_size=0.2,
                saved=True)


if __name__ == '__main__':
    root = "F:/code/python/EANN-KDD18-degugged11.2/test/pairdata"
    run_eann(root, False)

    # image_transforms = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    #
    # word_vectors, word_idx_map, max_text_len = build_word2vec(root)
    # word_vectors, word_idx_map = padding_vec_and_idx(
    #     word_vectors, word_idx_map)
    # print(f'using corpus from {root} to train word2vec')
    #
    # embedding_params = {
    #     'word_idx_map': word_idx_map,
    #     'max_text_len': max_text_len
    # }
    #
    # max_text_len, event_labels = generate_max_text_len_and_event_label(root)
    # event_num = max(event_labels) + 1
    #
    # dataset = FolderMultiModalDataset(root,
    #                                   embedding=eann_word2idx,
    #                                   transform=image_transforms,
    #                                   embedding_params=embedding_params,
    #                                   event_label=torch.tensor(event_labels))
    # # for text, image, other_data, label in dataset:
    # #     print(text[0], text[1], image, other_data['event_label'], label, sep='\n\n')
    # #     break
    # #
    # model = EANN(event_num,
    #              hidden_size=32,
    #              dropout=1,
    #              reverse_lambd=1,
    #              vocab_size=len(word_idx_map),
    #              embed_weight=word_vectors)
    # loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
    #                                     list(model.parameters())),
    #                              lr=0.0003)
    # evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])
    #
    # trainer = EANNTrainer(model, evaluator, optimizer)
    # trainer.fit(dataset, batch_size=20, epochs=100, validate_size=0.2, saved=True)
