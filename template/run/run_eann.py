import os
import pickle as pickle
from typing import Dict, List, Any

import numpy as np
import torch
from torchvision import transforms

from data.process.text_process import build_word2vec, padding_vec_and_idx
from evaluate.evaluator import Evaluator
from model.multi_modal.eann import EANN
from template.data.dataset.multi_modal_dataset import FolderMultiModalDataset
from template.data.process.text_process import tokenize
from train.eann_trainer import EANNTrainer


def generate_post_event_map() -> Dict:
    id_files = ['train', 'test', 'validate']
    post_event_map = {}
    for id_file in id_files:
        id_ = pickle.load(
            open(
                "F:/code/python/EANN-KDD18-degugged11.2/Data/weibo/" + id_file + "_id.pickle",
                'rb'))
        post_event_map.update(id_)
    return post_event_map


# todo 将article_event_map中event id来代替event label
def generate_event_label(post_id: str, post_event_map: Dict[str, int], event_label_map: Dict[int, int]) -> int:
    event = int(post_event_map[post_id])
    if event not in event_label_map:
        event_label_map[event] = len(event_label_map)
    event_label = event_label_map[event]
    return event_label


def generate_mask(words_len: int, max_text_len: int):
    mask = torch.zeros(max_text_len, dtype=torch.int)
    mask[:words_len] = 1
    return mask


def generate_mask_and_event_label(path: str, max_text_len: int):
    event_labels = []
    masks = []
    event_label_map = {}
    post_event_map = generate_post_event_map()

    for dir in os.listdir(path):
        for entry in os.scandir(os.path.join(path, dir)):
            # 找到路径下的文本文件
            if os.path.splitext(entry.name)[1] == ".txt":
                file_path = os.path.join(path, dir, entry.name)
                with open(file_path, encoding='utf-8') as f:
                    for i, line in enumerate(f.readlines()):
                        line = line.rstrip()

                        # 第1行 post_id
                        if (i + 1) % 2 == 1:
                            post_id = line

                        # 第2行 文本内容
                        if (i + 1) % 2 == 0:
                            tokens = tokenize(line)
                            if post_id in post_event_map:
                                event_labels.append(
                                    generate_event_label(post_id, post_event_map, event_label_map))
                                masks.append(generate_mask(len(tokens), max_text_len))
    return torch.stack(masks), event_labels


def word_to_idx(text: List[str], word_idx_map: Dict[str, int], max_text_len: int):
    """convert words in text to id"""
    # todo 优化循环
    words_id = [word_idx_map[word] for word in text]  # 把每个word转为对应id
    while len(words_id) < max_text_len:  # 填充0
        words_id.append(0)
    return words_id


def eann_word2vec(path: str, other_params: Dict[str, Any]) -> torch.Tensor:
    with open(path, encoding='utf-8') as f:
        _ = f.readline()
        line = f.readline()  # 第二行才是文本
        tokens = tokenize(line)
    word_vectors, word_idx_map, max_text_len = other_params['word_vectors'], other_params['word_idx_map'], other_params[
        'max_text_len']
    words_id = word_to_idx(tokens, word_idx_map, max_text_len)
    embedding_result = word_vectors[words_id]
    if not isinstance(embedding_result, torch.Tensor):
        embedding_result = torch.tensor(embedding_result, dtype=torch.float64)
    return embedding_result


def eann_word2idx(path: str, other_params: Dict[str, Any]) -> torch.Tensor:
    with open(path, encoding='utf-8') as f:
        _ = f.readline()
        line = f.readline()  # 第二行才是文本
        tokens = tokenize(line)
    word_idx_map, max_text_len = other_params['word_idx_map'], other_params['max_text_len']
    words_id = word_to_idx(tokens, word_idx_map, max_text_len)
    return torch.tensor(words_id)


def run_eann(root: str, pre_trained_word2vec=True, max_text_len: int = None, word_vectors: np.ndarray = None,
             word_idx_map: Dict[str, int] = None):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not pre_trained_word2vec:
        print(f'using corpus from {root} to train word2vec')
        word_vectors, word_idx_map, max_text_len = build_word2vec(root)
        word_vectors, word_idx_map = padding_vec_and_idx(word_vectors, word_idx_map)
    masks, event_labels = generate_mask_and_event_label(root, max_text_len)
    event_num = max(event_labels) + 1

    embedding_params = {'word_vectors': word_vectors, 'word_idx_map': word_idx_map, 'max_text_len': max_text_len}

    dataset = FolderMultiModalDataset(root, embedding=eann_word2vec, transform=image_transforms,
                                      mask=masks, event_label=torch.tensor(event_labels),
                                      embedding_params=embedding_params)
    # model = EANN(event_num,
    #              len(word_idx_map),
    #              hidden_size=32,
    #              dropout=1,
    #              reverse_lambd=1,
    #              embed_weight=word_vectors)
    # criterion = torch.nn.CrossEntropyLoss()
    model = EANN(event_num,
                 hidden_size=32,
                 embed_dim=word_vectors.shape[0],
                 dropout=1,
                 reverse_lambd=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=0.0003)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = EANNTrainer(model, evaluator, criterion, optimizer)
    trainer.fit(dataset, batch_size=20, epochs=100, validate_size=0.2, saved=True)


if __name__ == '__main__':
    root = "F:/code/python/EANN-KDD18-degugged11.2/test/pairdata"
    text_path = 'F:/code/python/EANN-KDD18-degugged11.2/test/pairdata/fake/3464577842643036.txt'

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    word_vectors, word_idx_map, max_text_len = build_word2vec(root, vector_size=32)
    word_vectors, word_idx_map = padding_vec_and_idx(word_vectors, word_idx_map)
    masks, event_labels = generate_mask_and_event_label(root, max_text_len)
    event_num = max(event_labels) + 1

    embedding_params = {'word_vectors': word_vectors, 'word_idx_map': word_idx_map, 'max_text_len': max_text_len}

    dataset = FolderMultiModalDataset(root, embedding=eann_word2idx, transform=image_transforms,
                                      embedding_params=embedding_params,
                                      mask=masks, event_label=torch.tensor(event_labels))

    # for text, image, label, other_data in dataset:
    #     print(text, image, label, other_data['mask'], other_data['event_label'], sep='\n\n')
    #     break

    model = EANN(event_num,
                 hidden_size=32,
                 embed_dim=word_vectors.shape[0],
                 dropout=1,
                 reverse_lambd=1,
                 vocab_size=len(word_idx_map),
                 embed_weight=word_vectors)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=0.0003)
    evaluator = Evaluator(['accuracy', 'precision', 'recall', 'f1'])

    trainer = EANNTrainer(model, evaluator, criterion, optimizer)
    trainer.fit(dataset, batch_size=20, epochs=100, validate_size=0.2, saved=True)
