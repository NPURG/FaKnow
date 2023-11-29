import re
from typing import Dict, List, Optional, Callable

import jieba
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer


def read_stop_words(path: str) -> List[str]:
    """
    Read stop words from a file.

    Args:
        path (str): The path to the file containing stop words.

    Returns:
        List[str]: A list of stop words.
    """
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = [str(line).strip() for line in f.readlines()]
    return stop_words


def chinese_tokenize(text: str,
                     stop_words: Optional[List[str]] = None) -> List[str]:
    """
    tokenize chinese text with jieba and regex to remove punctuation

    Args:
        text (str): text to be tokenized
        stop_words (List[str]): stop words, default=None

    Returns:
        List[str]: tokenized text
    """

    cleaned_text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "",
                          text).strip().lower()

    split_words = jieba.lcut(cleaned_text)
    if stop_words is None:
        return split_words
    return [word for word in split_words if word not in stop_words]


def english_tokenize(text: str) -> List[str]:
    """
    tokenize english text with nltk and regex to remove punctuation

    Args:
        text (str): text to be tokenized

    Returns:
        List[str]: tokenized text
    """

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, ' ', text)

    text = text.strip()
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    # porter stemmer and lemmatizer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


class TokenizerFromPreTrained:
    """
    Tokenizer for pre-trained models in transformers with fixed length,
    return token_id and mask
    """
    def __init__(self,
                 max_len: int,
                 bert: str,
                 text_preprocessing: Optional[Callable[[List[str]],
                                                       List[str]]] = None):
        """
        Args:
            max_len (int): max length of input text
            bert (str): bert model name
            text_preprocessing (Optional[Callable[[List[str]], List[str]]]):
                text preprocessing function. Defaults to None.
        """

        self.text_preprocessing = text_preprocessing
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(bert)

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        tokenize texts

        Args:
            texts (List[str]): texts to be tokenized

        Returns:
            Dict[str, torch.Tensor]: tokenized texts
                with key 'token_id' and 'mask'
        """
        if self.text_preprocessing is not None:
            texts = self.text_preprocessing(texts)

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
