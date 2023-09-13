import  torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import *
import torchvision

from transformers import BertModel
from faknow.model.model import AbstractModel
from faknow.model.layers.transformer import AddNorm, MultiHeadAttention, FFN, PositionalEncoding

class HMCAN(AbstractModel):
    r"""
    HMCAN: Hierarchical Multi-modal Contextual Attention Network for fake news Detection, SIGIR 2021
    paper: https://arxiv.org/abs/2103.00113
    code: https://github.com/wangjinguang502/HMCAN
    """
    def __init__(self,
                 word_max_length = 20,
                 left_num_layers = 2,
                 left_num_heads = 12,
                 dropout = 0.1,
                 right_num_layers = 2,
                 right_num_heads = 12,
                 alpha = 0.7):
        """

        Args:
            word_max_length(int): the max length of input, Default=20.
            left_num_layers(int): the numbers of  the left Attention&FFN layer in the Contextual Transformer, Default=2.
            left_num_heads(int): the numbers of head in the Multi-Head Attention layer(in the left Attention&FFN), Default=12.
            dropout(float): dropout rate, Default=0.1.
            right_num_layers(int): the numbers of  the right Attention&FFN layer in the Contextual Transformer, Default=2.
            right_num_heads(int): the numbers of head in the Multi-Head Attention layer(in the right Attention&FFN), Default=12.
            alpha(float): the weight of the first Attention&FFN layer's output, Default=0.7.
        """

        super(HMCAN, self).__init__()
        self.word_length = word_max_length
        self.alpha = alpha
        self.output_dims = 768
        self.loss_func = nn.CrossEntropyLoss()
        # text
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states = True,
                                              output_attentions =True).requires_grad_(False)

        # image
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet50.parameters():
            param.requires_grad = False
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-2])
        self.image_conv = nn.Conv2d(2048, 768, 4)
        self.image_bn = nn.BatchNorm2d(768)

        # Contextual Transformer
        self.contextual_transform1 = TextImage_Transformer(left_num_layers, left_num_heads,
                                                          right_num_layers,right_num_heads,
                                                          dropout, self.output_dims)
        self.contextual_transform2 = TextImage_Transformer(left_num_layers, left_num_heads,
                                                          right_num_layers, right_num_heads,
                                                          dropout, self.output_dims)

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(768*6, 256),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 2))

    def forward(self, token_id: torch.Tensor,
                mask: torch.Tensor, image: torch.Tensor):
        """

         Args:
            token_id (Tensor): text token ids
            image (Tensor): image pixels
            mask (torch.Tensor): text masks

        Returns:
            class_output (Tensor): prediction of being fake news, shape=(batch_size, 2)
        """

        mask = torch.ones_like(mask) # ban mask

        semantics = self.bert(token_id,
                              attention_mask=mask).hidden_states[1:]  # extract features from all the 12 block in bert-base model
        text_embeding = []
        for i in range(3):
            text_excerpt = semantics[0 + i] + semantics[1 + i] + semantics[2 + i] + semantics[3 + i]
            text_embeding.append(text_excerpt)

        image_features = self.resnet50(image)
        image_features = F.relu(self.image_bn(self.image_conv(image_features))) # [batch_size, 768, 4, 4]
        image_features = image_features.view(image_features.shape[0], image_features.shape[1], -1)
        image_features = image_features.permute(0, 2, 1)  # [batch_size, 16, 768]

        output = []
        for i in range(3):
            text_image = self.contextual_transform1(text_embeding[i], mask, image_features, None)
            image_text = self.contextual_transform2(image_features, None, text_embeding[i], mask)
            output_feature = self.alpha * text_image + (1 - self.alpha) * image_text
            output.append(output_feature)

        classifier_input = torch.cat((output[0], output[1], output[2]), dim=1)
        classifier_output = self.classifier(classifier_input)

        return classifier_output

    def calculate_loss(self, data: Dict[str, Any]) -> Tensor:
        """
         calculate total loss

         Args:
            data(Dict[str, any]): batch data dict

        Returns:
            Tensor: total_loss
        """
        token_id = data['text']['token_id']
        mask = data['text']['mask']
        image = data['image']
        label = data['label']
        output = self.forward(token_id, mask, image)
        loss = self.loss_func(output, label)

        return loss

    def predict(self, data_without_label: Dict[str, Any]) -> Tensor:
        """
            predict the probability of being fake news

            Args:
                data_without_label (Dict[str, Any]): batch data dict

            Returns:
                Tensor: softmax probability, shape=(batch_size, 2)
         """
        token_id = data_without_label['text']['token_id']
        mask = data_without_label['text']['mask']
        image = data_without_label['image']
        pred = self.forward(token_id, mask, image)
        pred = torch.softmax(pred, dim=-1)
        return pred

class TextImage_Transformer(nn.Module):
    """

    Contextual Attention Network of combining image features with text feature
    """
    def __init__(self,left_num_layers: int, left_num_heads: int,
                 right_num_layers: int, right_num_heads: int,
                 dropout: float, feature_dim: int):
        """

        left_num_layers(int): the num of layer for the left transformer block.
        left_num_heads(int): the numbers of heads in the left multiheadattention layer.
        right_num_layers(int): the num of layer for the right transformer block.
        right_num_heads(int): the numbers of heads in the right multiheadattention layer.
        dropout(float): dropout rate.
        feature_dim(int): the feature dimension of input.
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        input_dim = feature_dim
        self.embedding = PositionalEncoding(input_dim, dropout, max_len=1000)

        self.transformer1 = TransformerEncoder(left_num_layers, input_dim, left_num_heads,
                                              input_dim, dropout)

        self.transformer2 = TransformerEncoder(right_num_layers, input_dim,
                                                   right_num_heads, input_dim,
                                                   dropout)

    def forward(self, left_features: Tensor, left_mask: Union[Tensor, None],
                right_features: Tensor, right_mask: Union[Tensor, None]):
        """

        left_features(Tensor): the left transformer's input, shape=[batch_size, length, embedding_dim].
        left_mask(Union[Tensor, None]): the mask of left input, shape=[batch_size, ...].
        right_features(Tensor): the right transformer's input, shape=[batch_size, length, embedding_dim].
        left_mask(Union[Tensor, None]): the mask of right input, shape=[batch_size, ...]
        """

        left_features = self.input_norm(left_features)
        left_features = self.embedding(left_features)
        left_features = self.transformer1(left_features, left_features, left_features, left_mask)
        left_pooled = torch.mean(left_features, dim=1)

        right_features = self.transformer2(right_features, left_features, left_features, right_mask)
        right_pooled = torch.mean(right_features, dim=1)

        return torch.cat([left_pooled, right_pooled], dim=-1)

class TransformerEncoder(nn.Module):
    """

    single Transformer block for TextImage_Transformer(Contextual Transformer)
    """
    def __init__(self, num_layers: int, input_dim: int, num_heads: int, feature_dims: int, dropout: float):
        """

        num_layer(int): the layer's nums of attention block.
        input_dim(int): input dimension.
        num_heads(int): the head's num of multihead attention.
        feature_dims(int): the feature dim of multihead attention passing to FFN.
        dropout(float): dropout rate.
        """
        super().__init__()
        self.input_dim = input_dim
        assert num_layers > 0
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(input_dim, feature_dims, num_heads, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Union[Tensor, None]):
        """

        query(Tensor): shape=(batch_size, q_num, d)
        key(Tensor): shape=(batch_size, k-v_num, d)
        value(Tensor): shape=(batch_size, k-v_num, v-dim)
        mask(Union[Tensor, None]): shape=(batch_size, ...)
        """
        if mask is not None:
            mask = mask.sum(-1, keepdim=False)
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)
        return sources

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ffn_hidden_size: int,
                 head_num: int,
                 dropout=0.,
                 bias=False):
        """
        Args:
            input_dim (int): input dimension
            ffn_hidden_size (int): hidden layer dimension of FFN
            head_num (int): number of attention heads
            dropout (float): dropout rate, default=0.
            bias (bool): whether to use bias in Linear layers, default=False
        """

        super(TransformerEncoderLayer, self).__init__()
        assert input_dim % head_num == 0,\
        f"model dim {input_dim} not divisible by {head_num} heads"

        self.attention = MultiHeadAttention(input_dim,
                                            input_dim,
                                            input_dim,
                                            head_num,
                                            out_size=input_dim // head_num,
                                            dropout=dropout,
                                            bias=bias)
        self.addnorm1 = AddNorm(input_dim, dropout)
        self.ffn = FFN(input_dim, ffn_hidden_size, input_dim, dropout, activation=nn.GELU())
        self.addnorm2 = AddNorm(input_dim, dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, valid_lens: Optional[Tensor] = None):
        """
        Args:
            query(Tensor): shape=(batch_size, num_steps, input_size)
            key(Tensor): shape=(batch_size, k-v_num, d)
            value(Tensor): shape=(batch_size, k-v_num, v-dim)
            valid_lens (Tensor): shape=(batch_size,), default=None

        Returns:
            Tensor: shape=(batch_size,) or (batch_size, q_num)
        """

        y = self.addnorm1(query, self.attention(query, key, value, valid_lens))
        return self.addnorm2(y, self.ffn(y))






