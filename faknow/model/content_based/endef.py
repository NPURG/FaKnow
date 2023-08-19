from typing import List, Optional, Dict, Callable, Union
from faknow.model.content_based import *

import torch
from torch import Tensor
from torch import nn
from transformers import BertModel
from faknow.model.content_based.mdfend import _MLP

from faknow.model.layers.layer import TextCNNLayer, CNNExtractor
from faknow.model.model import AbstractModel

class CNNExtractor(nn.Module):
    def __init__(self,
                 filter_num : int,
                 fliter_sizes: List[int],
                 input_size: int):
        """

        Args:
            filter_num(int): numbers of each kernel
            filter_size(List[int]): the list of kernel size
            input_size(int) : size of input's dimension
        """
        super(CNNExtractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [nn.Conv1d(input_size, filter_num, filter_size)
             for filter_size in fliter_sizes])

    def forward(self, input_data: torch.Tensor):
        """
        Args:
            input_data(torch.Tensor): shape=(batch_size, max_len, embedding_size)
        """
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

def dict_to_dict(inputs: Dict):
    """
    change inputs to one layer dict if it's nesting
    """

    innerdict_keep = {}
    keys_keep = []
    for keys, values in inputs.items():
        if type(inputs[keys]) == dict:
            keys_keep.append(keys)
            for key, value in inputs[keys].items():
                innerdict_keep[key] = value

    for keys, values in innerdict_keep.items():
        inputs[keys] = values

    for keys in keys_keep:
        del inputs[keys]

class ENDEF(AbstractModel):
    r"""
    ENDEF: Generalizing to the Future: Mitigating Entity Bias in Fake News Detection，SIGIR 2022
    paper: https://arxiv.org/pdf/2204.09484.pdf
    code: https://github.com/ICTMCG/ENDEF-SIGIR2022
    """

    def __init__(self,
                 pre_trained_bert_name: str,
                 base_model: Callable,
                 mlp_dims: Optional[List[int]] = None,
                 dropout_rate = 0.2,
                 entity_weight = 0.1):
        """

        Args:
            pre_trained_bert_name(str): the name or local path of pre-trained bert model
            base_model(Callable): the base model(content_based) using with entity features
            mlp_dims(List[int]): a list of the dimensions in MLP layer, if None, [384] will be taken as default
            entity_weight(float): the weight of entity， formula: (1-entity_weight) * bias_prediction + entity_weight * entity_prediction
        """
        super(ENDEF, self).__init__()
        self.loss_func = nn.BCELoss()
        self.bert = BertModel.from_pretrained(
            pre_trained_bert_name).requires_grad_(False)

        self.base_model = base_model
        self.entity_weight = entity_weight
        self.embedding_size = self.bert.config.hidden_size

        if mlp_dims is None:
            mlp_dims = [384]

        filter_num = 64
        filter_sizes = [1, 2, 3, 5, 10]

        self.entity_convs = CNNExtractor(filter_num, filter_sizes, self.embedding_size)
        mlp_input_shape = sum([filter_num for filter_size in filter_sizes])
        self.entity_mlp = _MLP(mlp_input_shape, mlp_dims, dropout_rate)
        self.entity_net = nn.Sequential(self.entity_convs, self.entity_mlp)

    def forward(self,
                base_model_params: Dict,
                entity_token_id: Tensor,
                entity_mask: Tensor):
        """

        Args:
            base_model_params(Dict): a dictionary including all param base_model.forward() need
            entity_token_id(Tensor): entity's token ids from bert tokenizer, shape=(batch, max_len)
            entity_mask(Tensor): mask from bert tokenizer, shape=(batch_size, max_len)

        Returns:
            FloatTensor: the prediction of being fake, shape = (batch_size, )
        """

        dict_to_dict(base_model_params)

        base_model_pred = self.base_model.forward(**base_model_params)

        entity_embedding = self.bert(entity_token_id,
                                     attention_mask=entity_mask)[0]

        entity_pred = self.entity_net(entity_embedding).squeeze(1)


        # 判断base_model的forward函数输出形式，选择不同输出形式
        if type(base_model_pred) is list:
            if base_model_pred[0].shape == entity_pred.shape:
                base_model_pred[0] = base_model_pred[0].unsqueeze(1)
            unbias_pred = (1 - self.entity_weight) * base_model_pred[0][:, -1] + self.entity_weight * entity_pred

        else:
            if base_model_pred.shape == entity_pred.shape:
                base_model_pred = base_model_pred.unsqueeze(1)
            unbias_pred = (1 - self.entity_weight) * base_model_pred[:, -1] + self.entity_weight * entity_pred

        if base_model_pred[:, -1].shape == torch.Size([2]):
            return torch.sigmoid(unbias_pred.squeeze(1)), torch.sigmoid(entity_pred), base_model_pred
        else:
            return torch.sigmoid(unbias_pred), torch.sigmoid(entity_pred), torch.sigmoid(base_model_pred)

    def calculate_loss(self, data, loss_weight=0.2) -> Tensor:
        """
        calculate loss via BCELoss

        Args:
            data (dict): batch data dict
            loss_weight(float): weight of prediction only considering entity

        Returns:
            loss (Tensor): loss value

        """

        entity_token_id = data['entity']['token_id']
        entity_mask = data['entity']['mask']
        label = data['label']
        del data['entity']
        del data['label']

        output, entity_output, _ = self.forward(data, entity_token_id, entity_mask)
        loss = self.loss_func(output, label.float()) + loss_weight * self.loss_func(entity_output, label.float())

        return loss

    def predict(self, data_without_label) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Dict[str, Any]): batch data dict

        Returns:
            Tensor: shape is same as base_model
        """
        return self.base_model.predict(data_without_label)





