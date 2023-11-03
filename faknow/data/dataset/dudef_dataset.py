import torch
import os
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset
import joblib
import pandas as pd
import jieba

save_dir = '../../../dataset/example/DUDEF/data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

label2idx = {'fake': 0, 'real': 1, 'unverified': 2}


baidu_emotions = ['angry', 'disgusting', 'fearful',
                  'happy', 'sad', 'neutral', 'pessimistic', 'optimistic']
baidu_emotions.sort()

baidu_emotions_2_index = dict(
    zip(baidu_emotions, [i for i in range(len(baidu_emotions))]))


def baidu_arr(emotions_dict=None):
    arr = np.zeros(len(baidu_emotions))

    if emotions_dict is None:
        return arr

    for k, v in emotions_dict.items():
        # like -> happy
        if k == 'like':
            arr[baidu_emotions_2_index['happy']] += v
        else:
            arr[baidu_emotions_2_index[k]] += v

    return arr


# load negation words
negation_words = []
with open('../../data/process/resources/Chinese/others/negative/negationWords.txt', 'r',encoding='utf-8') as src:
    lines = src.readlines()
    for line in lines:
        negation_words.append(line.strip())

# load degree words
how_words_dict = dict()
with open('../../data/process/resources/Chinese/HowNet/intensifierWords.txt', 'r',encoding='utf-8') as src:
    lines = src.readlines()
    for line in lines:
        how_word = line.strip().split()
        how_words_dict[' '.join(how_word[:-1])] = float(how_word[-1])

# negation value and degree value
def get_not_and_how_value(cut_words, i, windows):
    not_cnt = 0
    how_v = 1

    left = 0 if (i - windows) < 0 else (i - windows)
    window_text = ' '.join(cut_words[left:i])

    for w in negation_words:
        if w in window_text:
            not_cnt += 1
    for w in how_words_dict.keys():
        if w in window_text:
            how_v *= how_words_dict[w]

    return (-1) ** not_cnt, how_v


_, words2array = joblib.load(
    '../../data/process/resources/Chinese/大连理工大学情感词汇本体库/preprocess/words2array_27351.pkl')


def dalianligong_arr(cut_words, windows=2):
    arr = np.zeros(29)

    for i, word in enumerate(cut_words):
        if word in words2array:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            arr += not_v * how_v * words2array[word]

    return arr


boson_words_dict = dict()
with open('../../data/process/resources/Chinese/BosonNLP/BosonNLP_sentiment_score.txt', 'r', encoding='utf-8') as src:
    lines = src.readlines()
    for line in lines:
        boson_word = line.strip().split()
        if len(boson_word) != 2:
            continue
        else:
            boson_words_dict[boson_word[0]] = float(boson_word[1])


def boson_value(cut_words, windows=2):
    value = 0

    for i, word in enumerate(cut_words):
        if word in boson_words_dict:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            value += not_v * how_v * boson_words_dict[word]

    return value


# Emoticon
emoticon_df = pd.read_csv(
    '../../data/process/resources/Chinese/others/emoticon/emoticon.csv')
emoticons = emoticon_df['emoticon'].tolist()
emoticon_types = list(set(emoticon_df['label'].tolist()))
emoticon_types.sort()
emoticon2label = dict(
    zip(emoticon_df['emoticon'].tolist(), emoticon_df['label'].tolist()))
emoticon2index = dict(
    zip(emoticon_types, [i for i in range(len(emoticon_types))]))


def emoticon_arr(text, cut_words):
    arr = np.zeros(len(emoticon_types))

    if len(cut_words) == 0:
        return arr

    for i, emoticon in enumerate(emoticons):
        if emoticon in text:
            arr[emoticon2index[emoticon2label[emoticon]]
                ] += text.count(emoticon)

    return arr / len(cut_words)


# Punctuation
def symbols_count(text):
    excl = (text.count('!') + text.count('！')) / len(text)
    ques = (text.count('?') + text.count('？')) / len(text)
    comma = (text.count(',') + text.count('，')) / len(text)
    dot = (text.count('.') + text.count('。')) / len(text)
    ellip = (text.count('..') + text.count('。。')) / len(text)

    return excl, ques, comma, dot, ellip


# Sentimental Words
def init_words(file):
    with open(file, 'r', encoding='utf-8') as src:
        words = src.readlines()
        words = [l.strip() for l in words]
    return list(set(words))


pos_words = init_words('../../data/process/resources/Chinese/HowNet/正面情感词语（中文）.txt')
pos_words += init_words('../../data/process/resources/Chinese/HowNet/正面评价词语（中文）.txt')
neg_words = init_words('../../data/process/resources/Chinese/HowNet/负面情感词语（中文）.txt')
neg_words += init_words('../../data/process/resources/Chinese/HowNet/负面评价词语（中文）.txt')

pos_words = set(pos_words)
neg_words = set(neg_words)

def sentiment_words_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0, 0]

    # positive and negative words
    sentiment = []
    for words in [pos_words, neg_words]:
        c = 0
        for word in words:
            if word in cut_words:
                c += 1
        sentiment.append(c)
    sentiment = [c / len(cut_words) for c in sentiment]

    # degree words
    degree = 0
    for word in how_words_dict:
        if word in cut_words:
            degree += how_words_dict[word]

    # negation words
    negation = 0
    for word in negation_words:
        negation += cut_words.count(word)
    negation /= len(cut_words)

    sentiment += [degree, negation]

    return sentiment

# Personal Pronoun
first_pronoun = init_words(
    '../../data/process/resources/Chinese/others/pronoun/1-personal-pronoun.txt')
second_pronoun = init_words(
    '../../data/process/resources/Chinese/others/pronoun/2-personal-pronoun.txt')
third_pronoun = init_words(
    '../../data/process/resources/Chinese/others/pronoun/3-personal-pronoun.txt')
pronoun_words = [first_pronoun, second_pronoun, third_pronoun]


def pronoun_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]

    pronoun = []
    for words in pronoun_words:
        c = 0
        for word in words:
            c += cut_words.count(word)
        pronoun.append(c)

    return [c / len(cut_words) for c in pronoun]


# Auxilary Features
def auxilary_features(text, cut_words):
    arr = np.zeros(17)

    arr[:5] = emoticon_arr(text, cut_words)
    arr[5:10] = symbols_count(text)
    arr[10:14] = sentiment_words_count(cut_words)
    arr[14:17] = pronoun_count(cut_words)

    return arr

def cut_words_from_text(text):
    return list(jieba.cut(text))


def extract_publisher_emotion(content, content_words, emotions_dict):
    text, cut_words = content, content_words

    arr = np.zeros(55)

    arr[:8] = baidu_arr(emotions_dict)
    arr[8:37] = dalianligong_arr(cut_words)
    arr[37:38] = boson_value(cut_words)
    arr[38:55] = auxilary_features(text, cut_words)

    return arr


def extract_social_emotion(comments, comments_words, mean_emotions_dict, max_emotions_dict):
    if len(comments) == 0:
        arr = np.zeros(55)
        mean_arr, max_arr = arr, arr
        return mean_arr, max_arr, np.concatenate([mean_arr, max_arr])

    arr = np.zeros((len(comments), 55))

    for i in range(len(comments)):
        arr[i] = extract_publisher_emotion(
            comments[i], comments_words[i], None)

    mean_arr = np.mean(arr, axis=0)
    max_arr = np.max(arr, axis=0)

    mean_arr[:8] = baidu_arr(mean_emotions_dict)
    max_arr[:8] = baidu_arr(max_emotions_dict)

    return mean_arr, max_arr, np.concatenate([mean_arr, max_arr])


def extract_dual_emotion(piece, COMMENTS=100):
    for k in ['content_emotions', 'comments100_emotions_mean_pooling', 'comments100_emotions_max_pooling']:
        if k not in piece:
            piece[k] = None

    publisher_emotion = extract_publisher_emotion(
        piece['content'], piece['content_words'], piece['content_emotions'])
    mean_arr, max_arr, social_emotion = extract_social_emotion(
        piece['comments'][:COMMENTS], piece['comments_words'][:COMMENTS], piece['comments100_emotions_mean_pooling'], piece['comments100_emotions_max_pooling'])
    emotion_gap = np.concatenate(
        [publisher_emotion - mean_arr, publisher_emotion - max_arr])

    dual_emotion = np.concatenate(
        [publisher_emotion, social_emotion, emotion_gap])
    return dual_emotion

def get_labels_arr(pieces):
    labels = torch.tensor([label2idx[p['label']] for p in pieces])
    return F.one_hot(labels)

class DudefDataset(Dataset):
    def __init__(self):
        super().__init__()

    def get_label(data_dir):
        label_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r',encoding='utf-8')) for t in ['train', 'val', 'test']]
        split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

        for t, pieces in split_datasets.items():
            labels_arr = get_labels_arr(pieces)
            np.save(os.path.join(label_dir, '{}_{}.npy'.format(
                t, labels_arr.shape)), labels_arr)

    def get_dualemotion(data_dir):

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
                for p in pieces:
                    p['content_words'] = cut_words_from_text(
                        p['content'])
                    p['comments_words'] = [cut_words_from_text(
                        com) for com in p['comments']]
                with open(os.path.join(save_dir, '{}.json'.format(t)), 'w', encoding='utf-8') as f:
                    json.dump(pieces, f, indent=4, ensure_ascii=False)

            emotion_arr = [extract_dual_emotion(
                p) for p in pieces]
            emotion_arr = np.array(emotion_arr)
            np.save(os.path.join(emotion_dir, '{}_{}.npy'.format(
                t, emotion_arr.shape)), emotion_arr)

    def get_senmantics(data_dir, MAX_NUM_WORDS,embeddings_index):
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

        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.random.randn(num_words, EMBEDDING_DIM)
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

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

        if 'semantics' in input_types:
            return train_data, val_data, test_data, train_label, val_label, test_label, data, label, semantics_embedding_matrix
        else:
            return train_data, val_data, test_data, train_label, val_label, test_label, data, label