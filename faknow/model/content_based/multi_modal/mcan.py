from typing import Optional, List

import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from transformers import BertModel
import torch.nn.functional as F

from faknow.model.layers.dct import DctStem, DctInceptionBlock, conv2d_bn_relu
from faknow.model.layers.transformer import FFN, AddNorm
from faknow.model.model import AbstractModel

"""
Multimodal Fusion with Co-Attention Networks for Fake News Detection
paper: https://aclanthology.org/2021.findings-acl.226/
code: https://github.com/wuyang45/MCAN_code
"""


class _VGG(nn.Module):
    def __init__(self):
        super(_VGG, self).__init__()
        vgg_19 = vgg19(weights=VGG19_Weights.DEFAULT)

        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])

    def forward(self, img):
        img = self.feature(img)
        img = img.view(img.size(0), -1)
        image = self.classifier(img)

        return image


class _DctCNN(nn.Module):
    def __init__(self,
                 dropout,
                 kernel_sizes,
                 num_channels,
                 in_channel=128,
                 branch1_channels=None,
                 branch2_channels=None,
                 branch3_channels=None,
                 branch4_channels=None,
                 out_channels=64):
        super(_DctCNN, self).__init__()

        if branch4_channels is None:
            branch4_channels = [32]
        if branch3_channels is None:
            branch3_channels = [64, 96, 96]
        if branch2_channels is None:
            branch2_channels = [48, 64]
        if branch1_channels is None:
            branch1_channels = [64]

        self.stem = DctStem(kernel_sizes, num_channels)

        self.InceptionBlock = DctInceptionBlock(
            in_channel,
            branch1_channels,
            branch2_channels,
            branch3_channels,
            branch4_channels,
        )

        self.maxPool = nn.MaxPool2d((1, 122))

        self.dropout = nn.Dropout(dropout)

        self.conv = conv2d_bn_relu(branch1_channels[-1] + branch2_channels[-1] +
                                   branch3_channels[-1] + branch4_channels[-1],
                                   out_channels,
                                   kernel_size=1)

    def forward(self, dct_img):
        dct_f = self.stem(dct_img)
        x = self.InceptionBlock(dct_f)
        x = self.maxPool(x)
        x = x.permute(0, 2, 1, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)

        x = x.reshape(-1, 4096)

        return x


class _ScaledDotProductAttention(nn.Module):

    def __init__(self, attention_dropout=0.5):
        super(_ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)

        return attention


class _MultiHeadAttention(nn.Module):
    """
    multi-head attention + add&norm
    """

    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(_MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = _ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        # 没有permute，错误写法，实际上相当于始终没有把head拆分开，一直都是拖着head * d作为最后一个维度
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        # scale 没有取对，变成了 dim_per_head // num_heads
        scale = (key.size(-1) // num_heads) ** -0.5
        attention = self.dot_product_attention(query, key, value,
                                               scale)

        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)

        # final linear projection W_o
        output = self.linear_final(attention).squeeze(-1)

        # add&norm
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        return output


class _CoAttentionLayer(nn.Module):
    """
    co-attention layer with 2 co-attention blocks
    """

    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(_CoAttentionLayer, self).__init__()

        self.attention_1 = _MultiHeadAttention(model_dim, num_heads, dropout)
        self.ffn1 = FFN(model_dim, ffn_dim, dropout=dropout)
        self.ffn_addnorm1 = AddNorm(model_dim, dropout)

        self.attention_2 = _MultiHeadAttention(model_dim, num_heads, dropout)
        self.ffn2 = FFN(model_dim, ffn_dim, dropout=dropout)
        self.ffn_addnorm2 = AddNorm(model_dim, dropout)

        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):
        """
        Args:
            image_output: shape=(batch_size, model_dim)
            text_output: shape=(batch_size, model_dim)

        Returns:
            fusion_output: shape=(batch_size, model_dim)
        """
        image_output = image_output.unsqueeze(-1)
        text_output = text_output.unsqueeze(-1)

        output1 = self.attention_1(image_output, text_output, text_output,
                                   attn_mask)
        output1 = self.ffn_addnorm1(output1, self.ffn1(output1))

        output2 = self.attention_2(text_output, image_output, image_output,
                                   attn_mask)
        output2 = self.ffn_addnorm2(output2, self.ffn2(output2))

        output = torch.cat([output1, output2], dim=1)
        output = self.fusion_linear(output)

        return output


class MCAN(AbstractModel):
    def __init__(self,
                 bert: str,
                 kernel_sizes: Optional[List[int]] = None,
                 num_channels: Optional[List[int]] = None,
                 model_dim=256,
                 drop_and_bn='drop-BN',
                 num_labels=2,
                 num_layers=1,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.5):

        super(MCAN, self).__init__()

        # check input
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        elif len(kernel_sizes) != 3 or not all(type(x) == int for x in kernel_sizes) or not all(
                x > 0 for x in kernel_sizes):
            raise ValueError("kernel_sizes must be a list of 3 positive integers")

        if num_channels is None:
            num_channels = [32, 64, 128]
        elif len(num_channels) != 3 or not all(type(x) == int for x in num_channels) or not all(
                x > 0 for x in num_channels):
            raise ValueError("num_channels must be a list of 3 positive integers")
        assert drop_and_bn in ['drop-BN', 'BN-drop', 'drop-only', 'BN-only', 'none'], \
            "drop_and_bn must be one of 'drop-BN', 'BN-drop', 'drop-only', 'BN-only', 'none'"

        self.model_dim = model_dim
        self.drop_and_bn = drop_and_bn

        self.bert = BertModel.from_pretrained(bert)
        self.linear_text = nn.Linear(self.bert.config.hidden_size, model_dim)
        self.bn_text = nn.BatchNorm1d(model_dim)

        self.dropout = nn.Dropout(dropout)

        # vgg image
        self.vgg = _VGG()
        self.linear_vgg = nn.Linear(4096, model_dim)
        self.bn_vgg = nn.BatchNorm1d(model_dim)

        # dct image
        self.dct_img = _DctCNN(
            dropout,
            kernel_sizes,
            num_channels,
            in_channel=128,
            branch1_channels=[64],
            branch2_channels=[48, 64],
            branch3_channels=[64, 96, 96],
            branch4_channels=[32],
            out_channels=64)
        self.linear_dct = nn.Linear(4096, model_dim)
        self.bn_dct = nn.BatchNorm1d(model_dim)

        # multimodal fusion
        self.fusion_layers = nn.ModuleList([
            _CoAttentionLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # classifier
        self.linear1 = nn.Linear(model_dim, 35)
        self.fc_bn = nn.BatchNorm1d(35)
        self.linear2 = nn.Linear(35, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def drop_bn_layer(self, x, part='dct'):
        bn = None
        if part == 'dct':
            bn = self.bn_dct
        elif part == 'vgg':
            bn = self.bn_vgg
        elif part == 'bert':
            bn = self.bn_text

        if self.drop_and_bn == 'drop-BN':
            x = self.dropout(x)
            x = bn(x)
        elif self.drop_and_bn == 'BN-drop':
            x = bn(x)
            x = self.dropout(x)
        elif self.drop_and_bn == 'drop-only':
            x = self.dropout(x)
        elif self.drop_and_bn == 'BN-only':
            x = bn(x)
        elif self.drop_and_bn == 'none':
            pass

        return x

    def forward(self, text_input_ids, token_type_ids, attention_mask, image,
                dct_img, attn_mask):

        # textual feature
        bert_output = self.bert(input_ids=text_input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        text_output = bert_output.pooler_output
        text_output = F.relu(self.linear_text(text_output))
        text_output = self.drop_bn_layer(text_output, part='bert')

        # vgg feature
        vgg_output = self.vgg(image)
        vgg_output = F.relu(self.linear_vgg(vgg_output))
        vgg_output = self.drop_bn_layer(vgg_output, part='vgg')

        # dct feature
        dct_out = self.dct_img(dct_img)
        dct_out = F.relu(self.linear_dct(dct_out))
        dct_out = self.drop_bn_layer(dct_out, part='dct')

        output = None
        for fusion_layer in self.fusion_layers:
            output = fusion_layer(vgg_output, dct_out, attn_mask)

        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, text_output, attn_mask)

        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)

        return output

    def calculate_loss(self, data):
        pass

    def predict(self, data):
        pass
