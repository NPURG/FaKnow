from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

from faknow.model.layers.layer import SignedGAT, TextCNNLayer
from faknow.model.layers.transformer import (FFN, ScaledDotProductAttention,
                                             transpose_qkv, transpose_output)
from faknow.model.model import AbstractModel
from faknow.utils.util import calculate_cos_matrix


class _TransformerBlock(nn.Module):
    """
    TransformerBlock for MFAN
    """
    def __init__(self,
                 input_size: int,
                 key_size=16,
                 value_size=16,
                 head_num=8,
                 dropout=0.1):
        super(_TransformerBlock, self).__init__()
        self.head_num = head_num
        self.k_size = key_size if key_size is not None else input_size
        self.v_size = value_size if value_size is not None else input_size

        # only for self-attention, the input dimensions of Q, K, V are the same
        self.W_q = nn.Parameter(torch.Tensor(input_size, head_num * key_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, head_num * key_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size,
                                             head_num * value_size))

        self.W_o = nn.Parameter(torch.Tensor(value_size * head_num,
                                             input_size))

        self.dropout = nn.Dropout(dropout)

        self.ffn = FFN(input_size, input_size, input_size, dropout)
        self.dot_product_attention = ScaledDotProductAttention(epsilon=1e-6,
                                                               dropout=dropout)
        self.__init_weights__()

    def __init_weights__(self):
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)
        nn.init.xavier_normal_(self.W_o)

        nn.init.xavier_normal_(self.ffn.dense1.weight)
        nn.init.xavier_normal_(self.ffn.dense2.weight)

    def multi_head_attention(self, Q, K, V):
        """
        Args:
            Q (Tensor): (batch_size, 1, embedding_size)
            K (Tensor): (batch_size, 1, embedding_size)
            V (Tensor): (batch_size, 1, embedding_size)
        """

        Q_ = transpose_qkv(Q.matmul(self.W_q), self.head_num)
        K_ = transpose_qkv(K.matmul(self.W_k), self.head_num)
        V_ = transpose_qkv(V.matmul(self.W_v), self.head_num)

        attention_score = self.dot_product_attention(Q_, K_, V_)
        attention_score = transpose_output(attention_score, self.head_num)

        output = self.dropout(attention_score.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        """
        only for self-attention, the input dimensions of Q, K, V are the same
        Args:
            Q (Tensor): (batch_size, 1, embedding_size)
            K (Tensor): (batch_size, 1, embedding_size)
            V (Tensor): (batch_size, 1, embedding_size)

        Returns:
            Tensor: output of transformer block, shape=(batch_size, 1, embedding_size)
        """

        attention_score = self.multi_head_attention(Q, K, V)
        # without norm
        X = Q + attention_score
        output = self.ffn(X) + X
        return output


class MFAN(AbstractModel):
    r"""
    MFAN: Multi-modal Feature-enhanced TransformerBlock Networks for Rumor Detection, IJCAI 2022
    paper: https://www.ijcai.org/proceedings/2022/335
    code: https://github.com/drivsaf/MFAN
    """
    def __init__(self,
                 word_vectors: torch.Tensor,
                 node_num: int,
                 node_embedding: torch.Tensor,
                 adj_matrix: torch.Tensor,
                 dropout_rate=0.6):
        """

        Args:
            word_vectors (Tensor): pretrained weights for word embedding
            node_num (int): number of nodes in graph
            node_embedding (Tensor): pretrained weights for node embedding
            adj_matrix (Tensor): adjacent matrix of graph
            dropout_rate (float): drop out rate. Default=0.6
        """
        super(MFAN, self).__init__()

        # text embedding
        self.word_embedding = nn.Embedding.from_pretrained(word_vectors,
                                                           freeze=False,
                                                           padding_idx=0)
        self.embedding_size = word_vectors.shape[-1]

        # text cnn
        kernel_sizes = [3, 4, 5]
        self.text_cnn_layer = TextCNNLayer(self.embedding_size, 100,
                                           kernel_sizes, nn.ReLU())

        # graph
        self.node_num = node_num
        self.cos_matrix = calculate_cos_matrix(node_embedding, node_embedding)
        self.signed_gat = SignedGAT(node_vectors=node_embedding,
                                    cos_sim_matrix=self.cos_matrix,
                                    num_features=self.embedding_size,
                                    node_num=self.node_num,
                                    head_num=1,
                                    adj_matrix=adj_matrix,
                                    dropout=0)

        # image
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, self.embedding_size)
        nn.init.eye_(self.resnet.fc.weight)

        # feature fusion
        self.align_graph = nn.Linear(self.embedding_size, self.embedding_size)
        self.align_text = nn.Linear(self.embedding_size, self.embedding_size)
        self.transformer_block = _TransformerBlock(
            input_size=self.embedding_size, head_num=8, dropout=0)

        # classification
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(nn.Linear(1800, 900),
                                        self.dropout, self.relu,
                                        nn.Linear(900, 600), self.relu,
                                        nn.Linear(600, self.embedding_size),
                                        self.relu, self.dropout,
                                        nn.Linear(self.embedding_size, 2))
        self.__init_weights__()

        self.loss_funcs = [nn.CrossEntropyLoss(), nn.MSELoss()]

    def __init_weights__(self):
        for module in self.classifier:
            if type(module) is torch.nn.Linear:
                nn.init.xavier_normal_(module.weight)

    def forward(self, post_id: torch.Tensor, text: torch.Tensor,
                image: torch.Tensor):
        """

        Args:
            post_id (Tensor): id of post, shape=(batch_size,)
            text (Tensor): token ids, shape=(batch_size, max_len)
            image (Tensor): shape=(batch_size, 3, width, height)

        Returns:
            tuple:
                - class_output (Tensor): prediction of being fake news, shape=(batch_size, 2)
                - dist (List[Tensor]): aligned text and aligned graph, shape=(batch_size, embedding_size)
        """
        image_feature = self.resnet(image).unsqueeze(1)

        graph_feature = self.signed_gat.forward(post_id).unsqueeze(1)

        text = self.word_embedding(text)
        text = self.text_cnn_layer(text)
        text_feature = text.unsqueeze(1)

        # text image graph: self-attention
        self_att_t = self.transformer_block(text_feature, text_feature,
                                            text_feature)
        self_att_g = self.transformer_block(graph_feature, graph_feature,
                                            graph_feature)
        self_att_i = self.transformer_block(image_feature, image_feature,
                                            image_feature)

        # text image: co-attention
        enhanced_text = self.transformer_block(self_att_i, self_att_t,
                                               self_att_t)

        # take enhanced text as new text feature
        self_att_t = enhanced_text

        # align graph and enhanced text via linear layer and get distance
        aligned_text = self.align_text(self_att_t.squeeze(1))
        aligned_graph = self.align_graph(self_att_g.squeeze(1))
        dist = [aligned_text, aligned_graph]

        # co attention
        co_att_tg = self.transformer_block(self_att_t, self_att_g,
                                           self_att_g).squeeze(1)
        co_att_gt = self.transformer_block(self_att_g, self_att_t,
                                           self_att_t).squeeze(1)
        co_att_ti = self.transformer_block(self_att_t, self_att_i,
                                           self_att_i).squeeze(1)
        co_att_it = self.transformer_block(self_att_i, self_att_t,
                                           self_att_t).squeeze(1)
        co_att_gi = self.transformer_block(self_att_g, self_att_i,
                                           self_att_i).squeeze(1)
        co_att_ig = self.transformer_block(self_att_i, self_att_g,
                                           self_att_g).squeeze(1)

        #  classification
        att_feature = torch.cat(
            (co_att_tg, co_att_gt, co_att_ti, co_att_it, co_att_gi, co_att_ig),
            dim=1)
        class_output = self.classifier(att_feature)

        return class_output, dist

    def calculate_loss(self, data) -> Dict[str, Tensor]:
        """
        calculate total loss, classification loss and distance loss,
        where total loss = classification loss + distance loss

        Args:
            data (Dict[str, Any]): batch data dict

        Returns:
            Dict[str, Tensor]: loss dict, key: total_loss, class_loss, dis_loss
        """

        post_id = data['post_id']
        text = data['text']
        image = data['image']
        label = data['label']
        class_output, dist = self.forward(post_id, text, image)
        class_loss = self.loss_funcs[0](class_output, label)
        dis_loss = self.loss_funcs[1](dist[0], dist[1])

        loss = class_loss + dis_loss

        return {
            'total_loss': loss,
            'class_loss': class_loss,
            'dis_loss': dis_loss
        }

    def predict(self, data_without_label) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Dict[str, Any]): batch data dict

        Returns:
            Tensor: softmax probability of being fake news, shape=(batch_size, 2)
        """

        post_id = data_without_label['post_id']
        text = data_without_label['text']
        image = data_without_label['image']
        class_output, _ = self.forward(post_id, text, image)
        return torch.softmax(class_output, dim=1)
