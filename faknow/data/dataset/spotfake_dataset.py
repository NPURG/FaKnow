import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import torch.nn.functional as F
from transformers import BertModel
import random
import time
import os
import re

# 预处理
def text_preprocessing(text):
    """
    - 删除实体@符号(如。“@united”)
    — 纠正错误(如:'&amp;' '&')
    @参数 text (str):要处理的字符串
    @返回 text (Str):已处理的字符串
    """
    # 去除 '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    #  替换'&amp;'成'&'
    text = re.sub(r'&amp;', '&', text)

    # 删除尾随空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class FakeNewsDataset(Dataset):

    def __init__(self, df, root_dir, image_transform, tokenizer, MAX_LEN):
        """
        参数:
            csv_file (string):包含文本和图像名称的csv文件的路径
            root_dir (string):目录
            transform(可选):图像变换
        """
        self.csv_data = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.tokenizer_bert = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]
    
    def pre_processing_BERT(self, sent):
        
        encoded_sent = self.tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  # 预处理
            add_special_tokens=True,        # `[CLS]`&`[SEP]`
            max_length=self.MAX_LEN,        # 截断/填充的最大长度
            padding='max_length',           # 句子填充最大长度
            # return_tensors='pt',          # 返回tensor
            return_attention_mask=True,     # 返回attention mask
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        # 转换tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
     
        
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.root_dir + self.csv_data['image_id'][idx] + '.jpg'
        image = Image.open(img_name).convert("RGB")
        image = self.image_transform(image)
        
        text = self.csv_data['post_text'][idx]
        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        label = self.csv_data['label'][idx]

        if label == 'fake':
            label = '1'
        else:
            label = '0'
        label = int(label)
        
        label = torch.tensor(label)

        sample = {
                  'image_id'  :  image, 
                  'BERT_ip'   : [tensor_input_id, tensor_input_mask],
                  'label'     :  label
                  }

        return sample