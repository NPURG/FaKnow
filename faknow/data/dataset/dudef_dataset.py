import torch
import os
import json
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Any

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import jieba


class DudefDataset(Dataset):
    def __init__(self, data_dir: str, baidu_arr: Callable,
                 dalianligong_arr: Callable, boson_value: Callable,
                 auxiliary_features: Callable):
        """
        Initialize the DudefDataset class.
        Args:
            baidu_arr (Callable): Function to calculate emotions using Baidu API.
            dalianligong_arr (Callable): Function to calculate emotions using Dalian Ligong University's method.
            boson_value (Callable): Function to extract boson features
            auxiliary_features (Callable): Function to extract auxiliary features from the text.
        """

        super().__init__()
        self.baidu_arr = baidu_arr
        self.dalianligong_arr = dalianligong_arr
        self.boson_value = boson_value
        self.auxiliary_features = auxiliary_features
        self.save_dir = os.path.join(data_dir, 'data')
        self.label2idx = {'fake': 0, 'real': 1, 'unverified': 2}

    def extract_publisher_emotion(self, content: str, content_words: List[str],
                                  emotions_dict: Optional[Dict] = None) -> Tensor:
        """
        Args:
            content (str): Original content of the publisher.
            content_words (List[str]): List of segmented words from the content.
            emotions_dict (Dict): Dictionary mapping emotions to their respective indices, default=None
        Returns:
            arr (Tensor): publisher emotion features, shape=(55,)
        """
        # Extract emotions from the publisher's content
        text, cut_words = content, content_words
        arr = torch.zeros(55)
        arr[:8] = self.baidu_arr(emotions_dict)
        arr[8:37] = self.dalianligong_arr(cut_words)
        arr[37:38] = self.dalianligong_arr(cut_words)
        arr[38:55] = self.auxiliary_features(text, cut_words)
        return arr

    def extract_social_emotion(self, comments: List[str],
                               comments_words: List[List[str]],
                               mean_emotions_dict: Dict,
                               max_emotions_dict: Dict) -> Tuple[Tensor, Tensor, Tensor]:
        """
           Extract emotions from social interactions (comments)
           Args:
               comments (List[str]): List of comments
               comments_words (List[List[str]]): List of words in comments
               mean_emotions_dict (Dict): Dictionary of mean emotions, comments100_emotions_mean_pooling
               max_emotions_dict (Dict): Dictionary of max emotions, comments100_emotions_max_pooling

           Returns:
               mean_arr (Tensor): Mean emotion array, shape=(55)
               max_arr (Tensor): Max emotion array, shape=(55)
               concatenated_arr (Tensor): Concatenated mean_arr and max_arr, shape=(110)
       """
        if len(comments) == 0:
            arr = torch.zeros(55)
            mean_arr, max_arr = arr, arr
            return mean_arr, max_arr, torch.cat([mean_arr, max_arr])

        arr = torch.zeros((len(comments), 55))
        for i in range(len(comments)):
            arr[i] = self.extract_publisher_emotion(comments[i],
                                                    comments_words[i], None)
        mean_arr = torch.mean(arr, dim=0)
        max_arr = torch.max(arr, dim=0)
        mean_arr[:8] = self.baidu_arr(mean_emotions_dict)
        max_arr[:8] = self.baidu_arr(max_emotions_dict)

        return mean_arr, max_arr, torch.cat([mean_arr, max_arr])

    def extract_dual_emotion(self, piece: Dict[str, Any], comments_num=100) -> Tensor:
        """
        Args:
             piece (Dict[str, Any]): Dictionary containing content information
             comments_num (int): Number of comments to consider, default=100
        Returns:
            dual_emotion (Tensor): Dual emotion array, shape=(275)
        """
        for k in [
            'content_emotions', 'comments100_emotions_mean_pooling',
            'comments100_emotions_max_pooling'
        ]:
            if k not in piece:
                piece[k] = None

        publisher_emotion = self.extract_publisher_emotion(
            piece['content'], piece['content_words'],
            piece['content_emotions'])
        mean_arr, max_arr, social_emotion = self.extract_social_emotion(
            piece['comments'][:comments_num], piece['comments_words'][:comments_num],
            piece['comments100_emotions_mean_pooling'],
            piece['comments100_emotions_max_pooling'])
        emotion_gap = torch.cat(
            [publisher_emotion - mean_arr, publisher_emotion - max_arr])

        dual_emotion = torch.cat(
            [publisher_emotion, social_emotion, emotion_gap])
        return dual_emotion

    def get_labels_arr(self, pieces: List) -> Tensor:
        """
        Args:
            pieces (List): List of pieces
        Returns:
            labels (Tensor): Tensor of labels,shape=(len(pieces), 2)
        """
        labels = torch.tensor([self.label2idx[p['label']] for p in pieces])
        return F.one_hot(labels)

    def get_label(self, data_dir: str):
        """
        Retrieves label arrays from the dataset and saves them.

        Args:
            data_dir (str): The directory containing the dataset.
        """
        label_dir = os.path.join(self.save_dir, 'labels')
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        split_datasets = [
            json.load(
                open(os.path.join(data_dir, '{}.json'.format(t)),
                     'r',
                     encoding='utf-8')) for t in ['train', 'val', 'test']
        ]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        for t, pieces in split_datasets.items():
            labels_arr = self.get_labels_arr(pieces)
            np.save(
                os.path.join(label_dir,
                             '{}_{}.npy'.format(t, labels_arr.shape)),
                labels_arr)

    def get_dual_emotion(self, data_dir: str):
        # Get dual emotion arrays from the dataset
        """
            Retrieves dual emotion arrays from the dataset.
            Args:
                data_dir (str): The directory containing the dataset.
        """
        emotion_dir = os.path.join(self.save_dir, 'emotions')
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)

        split_datasets = [
            json.load(
                open(os.path.join(data_dir, '{}.json'.format(t)),
                     'r',
                     encoding='utf-8')) for t in ['train', 'val', 'test']
        ]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        for t, pieces in split_datasets.items():
            arr_is_saved = False
            json_is_saved = False
            for j in os.listdir(os.path.join(self.save_dir, 'emotions')):
                if '.npy' in j and t in j:
                    arr_is_saved = True
            for f in os.listdir(self.save_dir):
                if t in f:
                    json_is_saved = True

            if arr_is_saved:
                continue

            if json_is_saved:
                pieces = json.load(
                    open(os.path.join(self.save_dir, '{}.json'.format(t)),
                         'r',
                         encoding='utf-8'))

            # words cutting
            if 'content_words' not in pieces[0].keys():
                for p in pieces:
                    p['content_words'] = list(jieba.cut(p['content']))
                    p['comments_words'] = list(
                        jieba.cut(com) for com in p['comments_num'])
                with open(os.path.join(self.save_dir, '{}.json'.format(t)),
                          'w',
                          encoding='utf-8') as f:
                    json.dump(pieces, f, indent=4, ensure_ascii=False)

            emotion_arr = torch.tensor([self.extract_dual_emotion(p) for p in pieces])
            torch.save(emotion_arr,
                os.path.join(emotion_dir,
                             '{}_{}.pt'.format(t, emotion_arr.shape)))

    def get_semantics(self, data_dir: str, max_num_words: int,
                      embeddings_index: Dict):
        # Get semantics arrays from the dataset
        """
            Retrieves semantics arrays from the dataset.
            Args:
                data_dir (str): The directory containing the dataset.
                max_num_words (int): The maximum number of words.
                embeddings_index (Dict): The embeddings index.
        """
        content_words_len = 100
        embedding_dim = 300

        output_dir = os.path.join(data_dir, 'semantics')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        split_datasets = [
            json.load(
                open(os.path.join(data_dir, '{}.json'.format(t)),
                     'r',
                     encoding='utf-8')) for t in ['train', 'val', 'test']
        ]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        texts = []
        for t in ['train', 'val', 'test']:
            texts += [' '.join(p['content_words']) for p in split_datasets[t]]

        tokenizer = get_tokenizer('basic_english')
        word_freq = {}
        for text in texts:
            tokens = tokenizer(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        sorted_words = sorted(word_freq.items(),
                              key=lambda x: x[1],
                              reverse=True)
        word_index = {
            word: index
            for index, (word, freq) in enumerate(sorted_words[:max_num_words])
        }

        sequences = []
        for text in texts:
            tokens = tokenizer(text)
            seq = [
                word_index[token] for token in tokens if token in word_index
            ]
            sequences.append(torch.tensor(seq))

        content_arr = pad_sequence(sequences,
                                   batch_first=True,
                                   padding_value=0)
        content_arr, b = content_arr.split(
            [content_words_len, len(content_arr[0]) - content_words_len], dim=1)

        num_words = min(max_num_words, len(word_index) + 1)
        embedding_matrix = np.random.randn(num_words, embedding_dim)
        for word, i in word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        np.save(
            os.path.join(
                output_dir,
                'embedding_matrix_{}.npy'.format(embedding_matrix.shape)),
            embedding_matrix)

        a, b = len(split_datasets['train']), len(split_datasets['val'])
        arrs = [content_arr[:a], content_arr[a:a + b], content_arr[a + b:]]
        for i, t in enumerate(['train', 'val', 'test']):
            np.save(
                os.path.join(output_dir, '{}_{}.npy'.format(t, arrs[i].shape)),
                arrs[i])

    def load_dataset(self, data_dir: str, batch_size: int, input_types: List):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset
            batch_size (int): num of batch size
            input_types (List): List of input types to be loaded.
        Returns:
            train_loader (DataLoader): train data
            val_loader (DataLoader): val data
            test_loader (DataLoader): test data
            semantics_embedding_matrix (Tensor): embedding matrix,shape=(6000,300)
        """
        # Load labels
        label_dir = os.path.join(data_dir, 'labels')
        for f in os.listdir(label_dir):
            f = os.path.join(label_dir, f)
            if 'train_' in f:
                train_label = np.load(f)
            elif 'val_' in f:
                val_label = np.load(f)
            elif 'test_' in f:
                test_label = np.load(f)
        # Load data
        train_data, val_data, test_data = [], [], []
        semantics_embedding_matrix = None
        for t in input_types:
            data_path = os.path.join(data_dir, t)
            for f in os.listdir(data_path):
                f = os.path.join(data_path, f)
                if 'train_' in f:
                    train_data.append(np.load(f))
                elif 'val_' in f:
                    val_data.append(np.load(f))
                elif 'test_' in f:
                    test_data.append(np.load(f))
                elif 'embedding_matrix_' in f:
                    semantics_embedding_matrix = np.load(f)

        if len(input_types) == 1:
            train_data, val_data, test_data = train_data[0], val_data[
                0], test_data[0]

        train_dataset = {}
        for i in range(len(train_data[0])):
            train_dataset[i] = {}
            train_dataset[i]['data'] = {}
            train_dataset[i]['data']['emotions'] = train_data[0][i]
            train_dataset[i]['data']['semantics'] = train_data[1][i]
            train_dataset[i]['label'] = train_label[i][1]
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        val_dataset = {}
        for i in range(len(val_data[0])):
            val_dataset[i] = {}
            val_dataset[i]['data'] = {}
            val_dataset[i]['data']['emotions'] = val_data[0][i]
            val_dataset[i]['data']['semantics'] = val_data[1][i]
            val_dataset[i]['label'] = val_label[i][1]
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True)

        test_dataset = {}
        for i in range(len(test_data[0])):
            test_dataset[i] = {}
            test_dataset[i]['data'] = {}
            test_dataset[i]['data']['emotions'] = test_data[0][i]
            test_dataset[i]['data']['semantics'] = test_data[1][i]
            test_dataset[i]['label'] = test_label[i][1]
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

        if 'semantics' in input_types:
            return train_loader, val_loader, test_loader, semantics_embedding_matrix
        else:
            return train_loader, val_loader, test_loader
