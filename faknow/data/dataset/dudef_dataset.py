import torch
import os
import sys
import json
from tqdm import tqdm
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from faknow.data.dataset.utils import load_embeddings
import torch.nn.functional as F
sys.path.append('../../data/dataset/')
import extract_emotion_ch
from torch.utils.data import Dataset

save_dir = '../../../dataset/example/DUDEF/data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

label2idx = {'fake': 0, 'real': 1, 'unverified': 2}


def get_labels_arr(pieces):
    labels = torch.tensor([label2idx[p['label']] for p in pieces])
    return F.one_hot(labels)

class DudefDataset(Dataset):
    def __init__(self):
        super().__init__()

    def get_label(data_dir):
        print('\n\n{} [{}]\tProcessing dataset label: {} \n'.format(
            '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '-'*20))

        label_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r',encoding='utf-8')) for t in ['train', 'val', 'test']]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        for t, pieces in split_datasets.items():
            labels_arr = get_labels_arr(pieces)
            print('{} dataset: got a {} label arr'.format(t, labels_arr.shape))
            np.save(os.path.join(label_dir, '{}_{}.npy'.format(
                t, labels_arr.shape)), labels_arr)

    def get_dualemotion(data_dir):
        print('\n\n{} [{}]\tProcessing the dataset dual_emotion: {}\n'.format(
            '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),'-'*20))

        extract_pkg = extract_emotion_ch

        emotion_dir = os.path.join(save_dir, 'emotions')
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in ['train', 'val', 'test']]
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
                pieces = json.load(
                    open(os.path.join(save_dir, '{}.json'.format(t)), 'r', encoding='utf-8'))

            # words cutting
            if 'content_words' not in pieces[0].keys():
                print('[{}]\tWords Cutting...'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
                for p in tqdm(pieces):
                    p['content_words'] = extract_pkg.cut_words_from_text(
                        p['content'])
                    p['comments_words'] = [extract_pkg.cut_words_from_text(
                        com) for com in p['comments']]
                with open(os.path.join(save_dir, '{}.json'.format(t)), 'w', encoding='utf-8') as f:
                    json.dump(pieces, f, indent=4, ensure_ascii=False)

            emotion_arr = [extract_pkg.extract_dual_emotion(
                p) for p in tqdm(pieces)]
            emotion_arr = np.array(emotion_arr)
            print('{} dataset: got a {} emotion arr'.format(t, emotion_arr.shape))
            np.save(os.path.join(emotion_dir, '{}_{}.npy'.format(
                t, emotion_arr.shape)), emotion_arr)

    def get_senmantics(data_dir, MAX_NUM_WORDS):
        print('\n\n{} [{}]\tProcessing the dataset senmantics: {}\n'.format(
            '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),'-'*20))

        embeddings_index = load_embeddings(
                language='Chinese', embeddings_file='../../data/process/resources/sgns.weibo.bigram-char')
        CONTENT_WORDS = 100
        EMBEDDING_DIM = 300

        output_dir = os.path.join(data_dir, 'semantics')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in ['train', 'val', 'test']]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        texts = []
        for t in ['train', 'val', 'test']:
            texts += [' '.join(p['content_words']) for p in split_datasets[t]]
        print('\n"dataset": {}, len(texts) = {}, \neg: texts[0] = {}\n'.format(
            sum([len(d) for d in split_datasets.values()]), len(texts), texts[0]))

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
        content_arr,b =content_arr.split([CONTENT_WORDS, len(content_arr[0])-CONTENT_WORDS], dim=1)
        print('Found {} unique tokens.'.format(len(word_index)))
        print('Content Array: {}'.format(content_arr.shape))

        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.random.randn(num_words, EMBEDDING_DIM)
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print('Embedding Matrix: {}'.format(embedding_matrix.shape))

        np.save(os.path.join(output_dir, 'embedding_matrix_{}.npy'.format(
            embedding_matrix.shape)), embedding_matrix)

        a, b = len(split_datasets['train']), len(split_datasets['val'])
        arrs = [content_arr[:a], content_arr[a:a+b], content_arr[a+b:]]
        for i, t in enumerate(['train', 'val', 'test']):
            np.save(os.path.join(output_dir, '{}_{}.npy'.format(
                t, arrs[i].shape)), arrs[i])


    def load_dataset(data_dir, input_types=['emotions']):
        label_dir = os.path.join(data_dir,'labels')
        for f in os.listdir(label_dir):
            f = os.path.join(label_dir, f)
            if 'train_' in f:
                train_label = np.load(f)
            elif 'val_' in f:
                val_label = np.load(f)
            elif 'test_' in f:
                test_label = np.load(f)

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

        print()
        for i, t in enumerate(['Train', 'Val', 'Test']):
            if len(input_types) == 1:
                print('{} data: {}, {} label: {}'.format(
                    t, data[i].shape, t, label[i].shape))
            else:
                print('{} data:'.format(t))
                for j, it in enumerate(input_types):
                    print('[{}]\t{}'.format(it, data[i][j].shape))
                print('{} label: {}\n'.format(t, label[i].shape))
        print()

        if 'semantics' in input_types:
            return train_data, val_data, test_data, train_label, val_label, test_label, data, label, semantics_embedding_matrix
        else:
            return train_data, val_data, test_data, train_label, val_label, test_label, data, label
