import random
import re
from typing import Dict, List

import numpy as np
import yaml
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.multi_modal.spotfake import SpotFake
from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str

__all__ = ['text_preprocessing', 'TokenizerSpotFake', 'transform_spotfake', 'run_spotfake', 'run_spotfake_from_yaml']


def text_preprocessing(text):
    """
    Preprocess the given text.

    - Remove entity '@' symbols (e.g., "@united")
    - Correct errors (e.g., '&amp;' to '&')

    Args:
        text (str): The text to be processed.

    Returns:
        str: The preprocessed text.
    """
    # 去除 '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    #  替换'&amp;'成'&'
    text = re.sub(r'&amp;', '&', text)

    # 删除尾随空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class TokenizerSpotFake:
    def __init__(self, max_len, pre_trained_bert_name):
        """
        Initialize the TokenizerSpotFake.

        Args:
            max_len (int): Maximum length for tokenization.
            pre_trained_bert_name (str): Name of the pre-trained BERT model.
        """
        self.max_len = max_len
        self.pre_trained_bert_name = BertTokenizer.from_pretrained(pre_trained_bert_name, do_lower_case=True)

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize the list of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the tokenized inputs.
        """
        # 定义列表存储文本处理后的结果
        input_ids_ls = []
        attention_mask_ls = []

        for text in texts:
            encoded_sent = self.pre_trained_bert_name.encode_plus(
                text=text_preprocessing(text),  # 预处理
                add_special_tokens=True,  # `[CLS]`&`[SEP]`
                max_length=self.max_len,  # 截断/填充的最大长度
                padding='max_length',  # 句子填充最大长度
                # return_tensors='pt',          # 返回tensor
                return_attention_mask=True,  # 返回attention mask
                truncation=True
            )
            input_ids = encoded_sent.get('input_ids')
            attention_mask = encoded_sent.get('attention_mask')

            # 转换tensor
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            # 添加到列表中去
            input_ids_ls.append(input_ids)
            attention_mask_ls.append(attention_mask)

        return {'input_ids': torch.stack(input_ids_ls), 'attention_mask': torch.stack(attention_mask_ls)}


def transform_spotfake(path: str) -> torch.Tensor:
    """
    Transform the image data at the given path.

    Args:
        path (str): Path to the image file.

    Returns:
        torch.Tensor: Transformed image data.
    """
    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(img)


def run_spotfake(
        train_path: str,
        validate_path: str = None,
        test_path: str = None,
        text_fc2_out: int = 32,
        text_fc1_out: int = 2742,
        dropout_p: float = 0.4,
        fine_tune_text_module: bool = False,
        img_fc1_out: int = 2742,
        img_fc2_out: int = 32,
        fine_tune_vis_module: bool = False,
        fusion_output_size: int = 35,
        loss_func=nn.BCELoss(),
        pre_trained_bert_name="bert-base-uncased",
        batch_size=8,
        epochs=50,
        max_len=500,
        lr=3e-5,
        metrics: List = None,
        device='cuda:0'
):
    """
    Train and evaluate the SpotFake model.

    Args:
        train_path (str): Path to the training data.
        validate_path (str, optional): Path to the validation data. Defaults to None.
        test_path (str, optional): Path to the test data. Defaults to None.
        text_fc2_out (int, optional): Output size for the text FC2 layer. Defaults to 32.
        text_fc1_out (int, optional): Output size for the text FC1 layer. Defaults to 2742.
        dropout_p (float, optional): Dropout probability. Defaults to 0.4.
        fine_tune_text_module (bool, optional): Fine-tune text module. Defaults to False.
        img_fc1_out (int, optional): Output size for the image FC1 layer. Defaults to 2742.
        img_fc2_out (int, optional): Output size for the image FC2 layer. Defaults to 32.
        fine_tune_vis_module (bool, optional): Fine-tune visual module. Defaults to False.
        fusion_output_size (int, optional): Output size for the fusion layer. Defaults to 35.
        loss_func (nn.Module, optional): Loss function. Defaults to nn.BCELoss().
        pre_trained_bert_name (str, optional): Name of the pre-trained BERT model. Defaults to "bert-base-uncased".
        batch_size (int, optional): Batch size. Defaults to 8.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        max_len (int, optional): Maximum length for tokenization. Defaults to 500.
        lr (float, optional): Learning rate. Defaults to 3e-5.
        metrics (List, optional): List of evaluation metrics. Defaults to None.
        device (str, optional): Device to run the training on ('cpu' or 'cuda'). Defaults to 'cuda:0'.
    """
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    tokenizer = TokenizerSpotFake(max_len, pre_trained_bert_name)

    training_set = MultiModalDataset(train_path, ['post_text'], tokenizer, ['image_id'], transform_spotfake)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    if validate_path is not None:
        validation_set = MultiModalDataset(validate_path, ['post_text'], tokenizer, ['image_id'], transform_spotfake)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    else:
        validation_loader = None

    model = SpotFake(text_fc2_out, text_fc1_out, dropout_p, fine_tune_text_module,
                     img_fc1_out, img_fc2_out, fine_tune_vis_module, fusion_output_size,
                     loss_func, pre_trained_bert_name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr
    )

    evaluator = Evaluator(metrics)

    trainer = BaseTrainer(model=model, evaluator=evaluator, optimizer=optimizer, device=device)
    trainer.fit(train_loader, epochs, validation_loader)

    if test_path is not None:
        test_set = MultiModalDataset(test_path, ['post_text'], tokenizer, ['image_id'], transform_spotfake)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        test_result = trainer.evaluate(test_loader)
        print(f"test result: {dict2str(test_result)}")


def run_spotfake_from_yaml(path: str):
    """
    Load SpotFake configuration from a YAML file and run the training and evaluation.

    Args:
        path (str): Path to the YAML configuration file.
    """
    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_spotfake(**_config)
