import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer


def df_filter(df_data, category_dict):
    # 用于根据给定的 category_dict 字典对输入的 DataFrame 进行过滤
    # 返回过滤后的 DataFrame，其中只包含了在 category_dict 中定义的类别的数据点。
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

def word2input(texts, max_len, dataset):
    if dataset == 'ch':
        tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    elif dataset == 'en':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

class M3FENDDataSet(Dataset):
    def __init__(self, path, max_len, category_dict, dataset):
        self.path = path
        self.max_len = max_len
        self.category_dict = category_dict
        self.dataset = dataset

        self.data = df_filter(read_pkl(self.path), self.category_dict)
        self.content = self.data['content'].to_numpy()
        self.comments = self.data['comments'].to_numpy()
        self.content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
        self.comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        self.emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        self.style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
        self.label = torch.tensor(self.data['label'].astype(int).to_numpy())
        self.category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        self.content_token_ids, self.content_masks = word2input(self.content, self.max_len, self.dataset)
        self.comments_token_ids, self.comments_masks = word2input(self.content, self.max_len, self.dataset)

    def __len__(self):
        return self.content_token_ids.size(0)

    def __getitem__(self, index):
        return {
            'content': self.content_token_ids[index],
            'content_masks': self.content_masks[index],
            'comments': self.comments_token_ids[index],
            'comments_masks': self.comments_masks[index],
            'content_emotion': self.content_emotion[index],
            'comments_emotion': self.comments_emotion[index],
            'emotion_gap': self.emotion_gap[index],
            'style_feature': self.style_feature[index],
            'category': self.category[index],
            'label': self.label[index]
        }