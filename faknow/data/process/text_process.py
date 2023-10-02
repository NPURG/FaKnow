import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import jieba
import numpy as np
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer

default_chinese_stop_words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_words',
                                               'stop_words.txt')


def get_stop_words(filepath=default_chinese_stop_words_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        stop_words = [str(line).strip() for line in f.readlines()]
    return stop_words


def chinese_tokenize(text,
                     stop_words: Optional[List[str]] = None,
                     stop_words_path=default_chinese_stop_words_path) -> List[str]:
    cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "",
                          text).strip().lower()

    split_words = jieba.lcut(cleaned_text)
    if stop_words is None:
        stop_words = get_stop_words(stop_words_path)
    return [word for word in split_words if word not in stop_words]


def english_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, ' ', text)

    text = text.strip()
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    # 提取词干
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # 同义词替换
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def generate_frequency_vocabulary(texts):
    vocab = defaultdict(int)
    for sentence in texts:
        for word in sentence:
            vocab[word] += 1
    return vocab


def add_unknown_words(word_vector_dict: Dict[str, np.ndarray], frequency_vocab: Dict[str, int], min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in frequency_vocab:
        if word not in word_vector_dict and frequency_vocab[word] >= min_df:
            word_vector_dict[word] = np.random.uniform(-0.25, 0.25, k)


def padding_vec_and_idx(word_vectors: np.ndarray, word_idx: Dict[str, int]):
    word_vectors = np.concatenate(
        [np.zeros((1, word_vectors.shape[1]), dtype='float32'), word_vectors])
    word_idx_map = {word: index + 1 for word, index in word_idx.items()}
    return word_vectors, word_idx_map


class TokenizerForBert:
    """
    Tokenizer for Bert with fixed length,
    return token_id and mask
    """

    def __init__(self, max_len: int, bert: str):
        """

        Args:
            max_len (int): max length of input text
            bert (str): bert model name
        """

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert)

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        tokenize texts

        Args:
            texts (List[str]): texts to be tokenized

        Returns:
            Dict[str, torch.Tensor]: tokenized texts
                with key 'token_id' and 'mask'
        """

        inputs = self.tokenizer(texts,
                                return_tensors='pt',
                                max_length=self.max_len,
                                add_special_tokens=True,
                                padding='max_length',
                                truncation=True)
        return {
            'token_id': inputs['input_ids'],
            'mask': inputs['attention_mask']
        }
