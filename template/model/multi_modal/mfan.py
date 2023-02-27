import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from template.model.layers.attention import (PositionWiseFFN,
                                             ScaledDotProductAttention)
from template.model.layers.layer import SignedGAT, TextCNNLayer
from template.model.model import AbstractModel
from template.utils.util import calculate_cos_matrix

"""
MFAN: Multi-modal Feature-enhanced TransformerBlock Networks for Rumor Detection
paper: https://www.ijcai.org/proceedings/2022/0335.pdf 
code: https://github.com/drivsaf/MFAN
"""


class TransformerBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 key_size=16,
                 value_size=16,
                 head_num=8,
                 attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        # input size，k，v size指的是X经过Linear变换后的得到的维度
        self.head_num = head_num
        self.k_size = key_size if key_size is not None else input_size
        self.v_size = value_size if value_size is not None else input_size

        self.W_q = nn.Parameter(torch.Tensor(input_size, head_num * key_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, head_num * key_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size,
                                             head_num * value_size))

        self.W_o = nn.Parameter(torch.Tensor(value_size * head_num,
                                             input_size))

        self.dropout = nn.Dropout(attn_dropout)

        self.ffn = PositionWiseFFN(input_size, input_size, input_size)
        self.dot_product_attention = ScaledDotProductAttention(
            epsilon=1e-6, dropout=attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)
        nn.init.xavier_normal_(self.W_o)

        nn.init.xavier_normal_(self.ffn.dense1.weight)
        nn.init.xavier_normal_(self.ffn.dense2.weight)

    def multi_head_attention(self, Q, K, V):
        # 64 * 1 * self.embedding_size，进去是什么形状，出来就是什么形状
        batch_size, q_len, _ = Q.size()
        batch_size, k_len, _ = K.size()
        batch_size, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(batch_size, q_len, self.head_num,
                                     self.k_size)
        K_ = K.matmul(self.W_k).view(batch_size, k_len, self.head_num,
                                     self.k_size)
        V_ = V.matmul(self.W_v).view(batch_size, v_len, self.head_num,
                                     self.v_size)

        Q_ = Q_.permute(0, 2, 1,
                        3).contiguous().view(batch_size * self.head_num, q_len,
                                             self.k_size)
        K_ = K_.permute(0, 2, 1,
                        3).contiguous().view(batch_size * self.head_num, q_len,
                                             self.k_size)
        V_ = V_.permute(0, 2, 1,
                        3).contiguous().view(batch_size * self.head_num, q_len,
                                             self.v_size)

        attention_socre = self.dot_product_attention(Q_, K_, V_)
        attention_socre = attention_socre.view(batch_size, self.head_num,
                                               q_len, self.v_size)
        attention_socre = attention_socre.permute(
            0, 2, 1, 3).contiguous().view(batch_size, q_len,
                                          self.head_num * self.v_size)

        output = self.dropout(attention_socre.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        attention_score = self.multi_head_attention(Q, K, V)
        # without norm
        X = Q + attention_score
        output = self.ffn(X) + X
        return output


class MFAN(AbstractModel):
    def __init__(self,
                 word_vectors: np.ndarray,
                 node_num: int,
                 node_embedding: np.ndarray,
                 adj_matrix: torch.Tensor,
                 dropout_rate=0.6):
        super(MFAN, self).__init__()

        # text embedding
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(word_vectors), freeze=False, padding_idx=0)
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
        self.transformer_block = TransformerBlock(
            input_size=self.embedding_size, head_num=8, attn_dropout=0)

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
        image_feature = self.resnet(image).unsqueeze(1)

        graph_feature = self.signed_gat.forward(post_id).unsqueeze(1)

        # text
        text = self.word_embedding(text)
        text = self.text_cnn_layer(text)
        text_feature = text.unsqueeze(1)

        # text image graph各自self-attention
        self_att_t = self.transformer_block(text_feature, text_feature,
                                            text_feature)
        self_att_g = self.transformer_block(graph_feature, graph_feature,
                                            graph_feature)
        self_att_i = self.transformer_block(image_feature, image_feature,
                                            image_feature)

        # 将Q换成image，text image co-attention，得到增强后的text
        enhanced_text = self.transformer_block(self_att_i, self_att_t,
                                               self_att_t).squeeze()

        # 此后使用的text，均为enhanced，而非最开始self-attention的版本
        self_att_t = enhanced_text.unsqueeze(1)

        # enhanced text与enhanced graph对齐
        aligned_text = self.align_text(enhanced_text)
        aligned_graph = self.align_graph(self_att_g)
        dist = [aligned_text, aligned_graph]

        # co attention
        co_att_tg = self.transformer_block(self_att_t, self_att_g,
                                           self_att_g).squeeze()
        co_att_gt = self.transformer_block(self_att_g, self_att_t,
                                           self_att_t).squeeze()
        co_att_ti = self.transformer_block(self_att_t, self_att_i,
                                           self_att_i).squeeze()
        co_att_it = self.transformer_block(self_att_i, self_att_t,
                                           self_att_t).squeeze()
        co_att_gi = self.transformer_block(self_att_g, self_att_i,
                                           self_att_i).squeeze()
        co_att_ig = self.transformer_block(self_att_i, self_att_g,
                                           self_att_g).squeeze()

        #  最终分类
        att_feature = torch.cat(
            (co_att_tg, co_att_gt, co_att_ti, co_att_it, co_att_gi, co_att_ig),
            dim=1)
        output = self.classifier(att_feature)

        return output, dist

    def calculate_loss(self, data):
        post_id, text, image, label = data
        class_output, dist = self.forward(post_id, text, image)
        class_loss = self.loss_funcs[0](class_output, label)
        dis_loss = self.loss_funcs[1](dist[0], dist[1].squeeze())

        loss = class_loss + dis_loss

        msg = f"class_loss={class_loss}, dis_loss={dis_loss}"

        return loss, msg

    def predict(self, data_without_label):
        post_id, text, image = data_without_label
        class_output, _ = self.forward(post_id, text, image)
        return class_output
