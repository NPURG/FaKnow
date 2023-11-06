import torch
import os
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset
import jieba
from faknow.data.dataset.text import TextDataset

# Define the save directory
save_dir = '../../../dataset/example/DUDEF/data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Define the label to index mapping
label2idx = {'fake': 0, 'real': 1, 'unverified': 2}


# Class for extracting emotions
class extract_emotion(TextDataset):
    def __init__(self, baidu_arr, dalianligong_arr, auxilary_features):
        """
            Initialize the ExtractEmotion class.

            Args:
                baidu_arr (function): Function to calculate emotions using Baidu API.
                dalianligong_arr (function): Function to calculate emotions using Dalian Ligong University's method.
                auxilary_features (function): Function to extract auxiliary features from the text.
        """
        super().__init__()
        self.baidu_arr = baidu_arr
        self.alianligong_arr = dalianligong_arr
        self.auxilary_features = auxilary_features

    def extract_publisher_emotion(self, content, content_words, emotions_dict):
        """
            Extract emotions from the publisher's content.

            Args:
                content (str): Original content of the publisher.
                content_words (list): List of segmented words from the content.
                emotions_dict (dict): Dictionary mapping emotions to their respective indices.

            Returns:
                numpy.ndarray: Array containing emotions extracted from the publisher's content.
        """
        # Extract emotions from the publisher's content
        text, cut_words = content, content_words

        arr = np.zeros(55)

        arr[:8] = self.baidu_arr(emotions_dict)
        arr[8:37] = self.dalianligong_arr(cut_words)
        arr[37:38] = self.dalianligong_arr(cut_words)
        arr[38:55] = self.auxilary_features(text, cut_words)

        return arr

    def extract_social_emotion(self, comments, comments_words, mean_emotions_dict, max_emotions_dict):
        # Extract emotions from social interactions (comments)
        """
           Extract emotions from social interactions (comments)

           Args:
               comments (list): List of comments
               comments_words (list): List of words in comments
               mean_emotions_dict (dict): Dictionary of mean emotions
               max_emotions_dict (dict): Dictionary of max emotions

           Returns:
               mean_arr (np.ndarray): Mean emotion array
               max_arr (np.ndarray): Max emotion array
               concatenated_arr (np.ndarray): Concatenated mean_arr and max_arr
       """
        if len(comments) == 0:
            arr = np.zeros(55)
            mean_arr, max_arr = arr, arr
            return mean_arr, max_arr, np.concatenate([mean_arr, max_arr])

        arr = np.zeros((len(comments), 55))

        for i in range(len(comments)):
            arr[i] = self.extract_publisher_emotion(comments[i], comments_words[i], None)

        mean_arr = np.mean(arr, axis=0)
        max_arr = np.max(arr, axis=0)

        mean_arr[:8] = self.baidu_arr(mean_emotions_dict)
        max_arr[:8] = self.baidu_arr(max_emotions_dict)

        return mean_arr, max_arr, np.concatenate([mean_arr, max_arr])

    def extract_dual_emotion(self, piece, COMMENTS=100):
        # Extract dual emotions from a piece of content
        """
            Extract dual emotions from a piece of content

            Args:
                piece (dict): Dictionary containing content information
                COMMENTS (int): Number of comments to consider (default=100)

            Returns:
                dual_emotion (np.ndarray): Dual emotion array
        """
        for k in ['content_emotions', 'comments100_emotions_mean_pooling', 'comments100_emotions_max_pooling']:
            if k not in piece:
                piece[k] = None

        publisher_emotion = self.extract_publisher_emotion(piece['content'], piece['content_words'],
                                                           piece['content_emotions'])
        mean_arr, max_arr, social_emotion = self.extract_social_emotion(piece['comments'][:COMMENTS],
                                                                        piece['comments_words'][:COMMENTS],
                                                                        piece['comments100_emotions_mean_pooling'],
                                                                        piece['comments100_emotions_max_pooling'])
        emotion_gap = np.concatenate([publisher_emotion - mean_arr, publisher_emotion - max_arr])

        dual_emotion = np.concatenate([publisher_emotion, social_emotion, emotion_gap])
        return dual_emotion

    def get_labels_arr(pieces):
        # Get labels array
        """
            Get labels array

            Args:
                pieces (list): List of pieces

            Returns:
                labels (torch.Tensor): Tensor of labels
        """
        labels = torch.tensor([label2idx[p['label']] for p in pieces])
        return F.one_hot(labels)


# Dataset class
class DudefDataset(Dataset):
    def __init__(self,baidu_arr, dalianligong_arr, auxilary_features):
        """
            Initializes a dataset object.

            Args:
                baidu_arr (torch.Tensor): The Baidu array.
                dalianligong_arr (torch.Tensor): The Dalianligong array.
                auxilary_features (torch.Tensor): The auxiliary features.
        """
        super().__init__()
        self.baidu_arr = baidu_arr
        self.dalianligong_arr = dalianligong_arr
        self.auxilary_features = auxilary_features

    def get_label(data_dir,baidu_arr, dalianligong_arr, auxilary_features):
        # Get label arrays from the dataset
        """
        Retrieves label arrays from the dataset.

        Args:
            data_dir (str): The directory containing the dataset.
            baidu_arr (torch.Tensor): The Baidu array.
            dalianligong_arr (torch.Tensor): The Dalianligong array.
            auxilary_features (torch.Tensor): The auxiliary features.
        """
        label_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in
                          ['train', 'val', 'test']]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        for t, pieces in split_datasets.items():
            labels_arr = extract_emotion.get_labels_arr(pieces,baidu_arr, dalianligong_arr, auxilary_features)
            np.save(os.path.join(label_dir, '{}_{}.npy'.format(t, labels_arr.shape)), labels_arr)

    def get_dualemotion(data_dir):
        # Get dual emotion arrays from the dataset
        """
            Retrieves dual emotion arrays from the dataset.

            Args:
                data_dir (str): The directory containing the dataset.
        """
        emotion_dir = os.path.join(save_dir, 'emotions')
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in
                          ['train', 'val', 'test']]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        for t, pieces in split_datasets.items():
            arr_is_saved = False
            json_is_saved = False
            for j in os.listdir(os.path.join(save_dir, 'emotions')):
                if '.npy' in j and t in j:
                    arr_is_saved = True
            for f in os.listdir(save_dir):
                if t in f:
                    json_is_saved = True

            if arr_is_saved:
                continue

            if json_is_saved:
                pieces = json.load(open(os.path.join(save_dir, '{}.json'.format(t)), 'r', encoding='utf-8'))

            # words cutting
            if 'content_words' not in pieces[0].keys():
                for p in pieces:
                    p['content_words'] = list(jieba.cut(p['content']))
                    p['comments_words'] = list(jieba.cut(com) for com in p['comments'])
                with open(os.path.join(save_dir, '{}.json'.format(t)), 'w', encoding='utf-8') as f:
                    json.dump(pieces, f, indent=4, ensure_ascii=False)

            emotion_arr = [extract_emotion.extract_dual_emotion(p) for p in pieces]
            emotion_arr = np.array(emotion_arr)
            np.save(os.path.join(emotion_dir, '{}_{}.npy'.format(t, emotion_arr.shape)), emotion_arr)

    def get_senmantics(data_dir, MAX_NUM_WORDS, embeddings_index):
        # Get semantics arrays from the dataset
        """
            Retrieves semantics arrays from the dataset.

            Args:
                data_dir (str): The directory containing the dataset.
                MAX_NUM_WORDS (int): The maximum number of words.
                embeddings_index: The embeddings index.
        """
        CONTENT_WORDS = 100
        EMBEDDING_DIM = 300

        output_dir = os.path.join(data_dir, 'semantics')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in
                          ['train', 'val', 'test']]
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

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        word_index = {word: index for index, (word, freq) in enumerate(sorted_words[:MAX_NUM_WORDS])}

        sequences = []
        for text in texts:
            tokens = tokenizer(text)
            seq = [word_index[token] for token in tokens if token in word_index]
            sequences.append(torch.tensor(seq))

        content_arr = pad_sequence(sequences, batch_first=True, padding_value=0)
        content_arr, b = content_arr.split([CONTENT_WORDS, len(content_arr[0]) - CONTENT_WORDS], dim=1)

        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.random.randn(num_words, EMBEDDING_DIM)
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        np.save(os.path.join(output_dir, 'embedding_matrix_{}.npy'.format(embedding_matrix.shape)), embedding_matrix)

        a, b = len(split_datasets['train']), len(split_datasets['val'])
        arrs = [content_arr[:a], content_arr[a:a + b], content_arr[a + b:]]
        for i, t in enumerate(['train', 'val', 'test']):
            np.save(os.path.join(output_dir, '{}_{}.npy'.format(t, arrs[i].shape)), arrs[i])

    def load_dataset(data_dir, input_types=['emotions']):
        """
          Load the dataset from the given directory.

          Args:
          - data_dir (str): Path to the directory containing the dataset.
          - input_types (list): List of input types to be loaded.
        """
        # Load labels
        label_dir = os.path.join(data_dir,'labels')
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
            train_data, val_data, test_data = train_data[0], val_data[0], test_data[0]

        data = [train_data, val_data, test_data]
        label = [train_label, val_label, test_label]
        # If only one input type is provided, unpack the arrays
        if 'semantics' in input_types:
            return train_data, val_data, test_data, train_label, val_label, test_label, data, label, semantics_embedding_matrix
        else:
            return train_data, val_data, test_data, train_label, val_label, test_label, data, label