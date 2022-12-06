import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import jieba
import numpy as np
from gensim.models import Word2Vec

default_chinese_stop_words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_words',
                                               'stop_words.txt')


def get_stop_words(filepath=default_chinese_stop_words_path):
    # print(os.path.relpath("F:/code/python/Template/template/data/process/stop_words/stop_words.txt"))
    with open(filepath, 'r', encoding='utf-8') as f:
        stop_words = [str(line).strip() for line in f.readlines()]
    return stop_words


def tokenize(text,
             stop_words: Optional[List[str]] = None,
             stop_words_path=default_chinese_stop_words_path) -> List[str]:
    cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "",
                          text).strip().lower()
    split_words = jieba.cut_for_search(cleaned_text)
    if stop_words is None:
        stop_words = get_stop_words(stop_words_path)
    # cleaned_text = " ".join([word for word in split_words if word not in stop_words])
    return [word for word in split_words if word not in stop_words]


def get_texts(root: str) -> Tuple[List[List[str]], int]:
    texts = []
    max_text_len = 0
    for dir in os.listdir(root):
        for entry in os.scandir(os.path.join(root, dir)):
            if os.path.splitext(entry.name)[1] == ".txt":
                file_path = os.path.join(root, dir, entry.name)
                with open(file_path, encoding='utf-8') as f:
                    for i, l in enumerate(f.readlines()):
                        l = l.rstrip()
                        if (i + 1) % 2 == 0:
                            tokens = tokenize(l)
                            if len(tokens) > max_text_len:
                                max_text_len = len(tokens)
                            texts.append(tokens)

    return texts, max_text_len


def generate_frequency_vocabulary(texts):
    vocab = defaultdict(int)
    for sentence in texts:
        for word in sentence:
            vocab[word] += 1
    return vocab


def add_unknown_words(word_vectors: np.ndarray, vocab: Dict, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vectors and vocab[word] >= min_df:
            word_vectors[word] = np.random.uniform(-0.25, 0.25, k)


def build_word2vec(root: str, min_count=1, vector_size=100, window=5):
    texts, max_text_len = get_texts(root)
    frequency_vocabulary = generate_frequency_vocabulary(texts)
    w2v = Word2Vec(texts,
                   min_count=min_count,
                   vector_size=vector_size,
                   window=window)
    add_unknown_words(w2v.wv, frequency_vocabulary)
    return w2v.wv.vectors, w2v.wv.key_to_index, max_text_len


def padding_vec_and_idx(word_vectors: np.ndarray, word_idx: Dict[str, int]):
    word_vectors = np.concatenate(
        [np.zeros((1, word_vectors.shape[1]), dtype='float32'), word_vectors])
    word_idx_map = {word: index + 1 for word, index in word_idx.items()}
    return word_vectors, word_idx_map
