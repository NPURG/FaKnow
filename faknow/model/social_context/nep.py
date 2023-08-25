from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BertModel

from faknow.model.model import AbstractModel


class _NewsEnvExtractor(nn.Module):
    def __init__(self,
                 bert_hidden_dim: int,
                 macro_env_out_dim=128,
                 micro_env_output_dim=128):
        """
        Args:
            bert_hidden_dim (int): bert hidden dimension
            macro_env_out_dim (int): macro environment output dimension, default=128
            micro_env_output_dim (int): micro environment output dimension, default=128
        """
        super(_NewsEnvExtractor, self).__init__()

        # gaussian kernel
        kernel_mu = torch.arange(-1, 1.1, 0.1).tolist()
        kernel_sigma = [20 for _ in kernel_mu]
        kernel_mu.append(0.99)
        kernel_sigma.append(100)
        self.kernel_mu = torch.tensor(kernel_mu)
        self.kernel_sigma = torch.tensor(kernel_sigma)

        # macro env
        self.macro_env_out_dim = macro_env_out_dim
        self.micro_env_output_dim = micro_env_output_dim

        macro_mlp_in = 0
        macro_mlp_in += 2 * bert_hidden_dim  # post与avg mac bert表示
        macro_mlp_in += len(self.kernel_mu)  # gaussian kernel

        self.macro_mlp = nn.Linear(macro_mlp_in, self.macro_env_out_dim)

        # micro env
        micro_output = 0
        # mic sem
        self.micro_sem_mlp = nn.Linear(2 * bert_hidden_dim,
                                       self.micro_env_output_dim)
        micro_output += self.micro_env_output_dim

        # mic sim
        self.micro_sim_mlp = nn.Linear(2 * len(self.kernel_mu),
                                       self.micro_env_output_dim)
        micro_output += self.micro_env_output_dim

        self.micro_mlp = nn.Linear(micro_output, self.micro_env_output_dim)

    def forward(self, post_simcse, avg_emac, avg_emic, kernel_p_emac,
                kernel_p_emic,
                kernel_avgmic_emic) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            post_simcse: post simcse representation, shape=(batch_size, simcse_dim)
            avg_emac: average macro news simcse representation, shape=(batch_size, simcse_dim)
            avg_emic: average micro news simcse representation, shape=(batch_size, simcse_dim)
            kernel_p_emac: gaussian kernel of similarity between post and macro news, shape=(batch_size, kernel_num)
            kernel_p_emic: gaussian kernel of similarity between  post and micro news, shape=(batch_size, kernel_num)
            kernel_avgmic_emic: gaussian kernel of similarity between average micro and micro news, shape=(batch_size, kernel_num)

        Returns:
            tuple:
                - macro_env: macro environment representation, shape=(batch_size, macro_env_out_dim)
                - micro_env: micro environment representation, shape=(batch_size, micro_env_out_dim)
        """
        p = post_simcse

        # ------------------ Macro Env ------------------
        # 宏观环境与post的高斯核，已事先计算好，只需norm
        kernel_p_emac = self.normalize(kernel_p_emac)

        # 三者concat作为最终宏观表征
        vectors = [x for x in [p, avg_emac, kernel_p_emac] if x is not None]
        v_p_mac = torch.cat(vectors, dim=-1)
        v_p_mac = self.macro_mlp(v_p_mac)

        # ------------------ Micro Env ------------------

        # mic sem
        # 平均值与post合并作为语义信息的聚合
        u_sem = torch.cat([p, avg_emic], dim=1)
        u_sem = self.micro_sem_mlp(u_sem)

        kernel_p_emic = self.normalize((kernel_p_emic))
        kernel_avgmic_emic = self.normalize(kernel_avgmic_emic)

        # mic sim
        u_sim = torch.cat([
            kernel_p_emic * kernel_avgmic_emic,
            kernel_p_emic - kernel_avgmic_emic
        ],
                          dim=1)
        u_sim = self.micro_sim_mlp(u_sim)

        vectors = [x for x in [u_sim, u_sem] if x is not None]
        v_p_mic = torch.cat(vectors, dim=-1)
        v_p_mic = self.micro_mlp(v_p_mic)

        return v_p_mac, v_p_mic

    def normalize(self, kernel_features):
        # Normalize
        zero = 1e-8
        kernel_sum = torch.sum(kernel_features)
        kernel_features /= (kernel_sum + zero)

        return kernel_features


class _BERT(nn.Module):
    def __init__(self,
                 bert: str,
                 max_len=256,
                 finetune_embedding_layers=True,
                 finetune_inter_layers=True):
        super(_BERT, self).__init__()

        self.bert = BertModel.from_pretrained(bert, return_dict=False)
        self.finetune(finetune_embedding_layers, finetune_inter_layers)

        self.max_len = max_len
        self.doc_max_len = self.max_len - 2
        self.out_dim = self.bert.config.hidden_size

    def finetune(self, finetune_embedding_layers: bool,
                 finetune_inter_layers: bool):
        for name, param in self.bert.named_parameters():
            # pooler layer
            if name.startswith("pooler"):
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data.normal_(mean=0.0,
                                       std=self.bert.config.initializer_range)
                param.requires_grad = True

            # last encoder layer
            elif name.startswith('encoder.layer.11'):
                param.requires_grad = True

            # embedding layer
            elif name.startswith('embeddings'):
                param.requires_grad = finetune_embedding_layers

            # the other transformer layers (intermediate layers)
            else:
                param.requires_grad = finetune_inter_layers

    def forward(self, token: torch.Tensor):
        # (batch_size, max_length)
        input_ids, masks = self._encode(token)

        # (batch_size, max_length, bert_dim)
        seq_output, _ = self.bert(input_ids)

        # (batch_size, bert_dim)
        output = torch.sum(masks * seq_output, dim=1)

        return output

    def _encode(self, tokens: torch.Tensor):
        input_ids = []
        masks = []
        for token in tokens:
            doc = token[:self.doc_max_len].tolist()

            # add CLS[101], SEP[102], PAD[103]
            # [101, ..., 102, 103, 103, ..., 103]
            padding_length = self.max_len - (len(doc) + 2)
            input_id = [101] + doc + [102] + [103] * padding_length
            input_ids.append(input_id)

            mask = torch.zeros(self.max_len, 1, dtype=torch.float)
            mask[:-padding_length] = 1 / (len(doc) + 2)
            masks.append(mask)

        return torch.tensor(input_ids, dtype=torch.long), torch.stack(masks)


class NEP(AbstractModel):
    """
    Zoom Out and Observe: News Environment Perception for Fake News Detection, ACL 2022
    paper: https://aclanthology.org/2022.acl-long.311/
    code: https://github.com/ICTMCG/News-Environment-Perception
    """

    def __init__(self,
                 macro_env_out_dim=128,
                 micro_env_out_dim=128,
                 num_mlp_layers=3,
                 fnd: nn.Module = _BERT,
                 fusion='gate',
                 attention_dim=128):
        """
        Args:
            macro_env_out_dim (int): dimension of macro environment representation, default=128
            micro_env_out_dim (int): dimension of micro environment representation, default=128
            num_mlp_layers (int): number of mlp layers, default=3
            fnd (nn.Module): fake news detector, default=_BERT
            fusion (str): fusion method to combine environment and fend, 'gate', 'att' or 'concat', default='gate'
            attention_dim (int): dimension of fusion attention layers, default=128
            **kwargs: parameters for fnd initialization
        """

        super(NEP, self).__init__()
        self.macro_env_out_dim = macro_env_out_dim
        self.micro_env_out_dim = micro_env_out_dim

        # === Env ===
        self.news_env_extractor = _NewsEnvExtractor(768, macro_env_out_dim,
                                                    micro_env_out_dim)

        # === FEND ===
        if fnd is not None:
            self.fake_news_detector = fnd
            last_output = self.fake_news_detector.out_dim

            # Env and FEND fusion
            self.fusion = fusion

            if self.fusion == 'att':
                self.macro_multihead_attn = nn.MultiheadAttention(
                    attention_dim, num_heads=8, dropout=0.5)
                self.micro_multihead_attn = nn.MultiheadAttention(
                    attention_dim, num_heads=8, dropout=0.5)
                last_output += 2 * attention_dim
            elif self.fusion == 'gate':
                assert self.macro_env_out_dim == self.micro_env_out_dim
                self.W_gate = nn.Linear(
                    self.fake_news_detector.out_dim + self.macro_env_out_dim,
                    self.macro_env_out_dim)
                last_output += self.macro_env_out_dim
            elif self.fusion == 'concat':
                last_output += self.macro_env_out_dim + self.micro_env_output_dim
            else:
                raise ValueError('Invalid fusion strategy.')
        else:
            self.fake_news_detector = None
            last_output = 0

        # === MLP layers ===
        self.fcs = []
        for _ in range(num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            self.fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output
        self.fcs.append(nn.Linear(last_output, 2))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, post_simcse, avg_mac, avg_mic, kernel_p_mac,
                kernel_p_mic, kernel_avg_mic_mic, **kwargs) -> Tensor:
        """
        Args:
            post_simcse: post simcse representation, shape=(batch_size, simcse_dim)
            avg_mac: average macro news simcse representation, shape=(batch_size, simcse_dim)
            avg_mic: average micro news simcse representation, shape=(batch_size, simcse_dim)
            kernel_p_mac: gaussian kernel of similarity between post and macro news, shape=(batch_size, kernel_num)
            kernel_p_mic: gaussian kernel of similarity between  post and micro news, shape=(batch_size, kernel_num)
            kernel_avg_mic_mic: gaussian kernel of similarity between average micro and micro news, shape=(batch_size, kernel_num)
            **kwargs: parameters for FND module forward

        Returns:
            Tensor: prediction of being fake, shape=(batch_size, 2)
        """

        # env
        v_p_mac, v_p_mic = self.news_env_extractor(post_simcse, avg_mac,
                                                   avg_mic, kernel_p_mac,
                                                   kernel_p_mic,
                                                   kernel_avg_mic_mic)

        if self.fake_news_detector is not None:
            # (bs, FEND_last_output)
            detector_output = self.fake_news_detector(**kwargs)

            # fusion
            output = None
            if self.fusion == 'concat':
                output = torch.cat([detector_output, v_p_mac, v_p_mic], dim=-1)
            elif self.fusion == 'att':
                output = self.forward_attention(detector_output, v_p_mac,
                                                v_p_mic)
            elif self.fusion == 'gate':
                output = self.forward_gate(detector_output, v_p_mac, v_p_mic)

        else:
            output = torch.cat([v_p_mac, v_p_mic], dim=-1)

        for fc in self.fcs:
            output = F.gelu(fc(output))

        return output

    def forward_attention(self,
                          detector_output=None,
                          v_p_mac=None,
                          v_p_mic=None):
        # (1, bs, 768)
        key = detector_output.unsqueeze(0)
        value = key

        # env_output: (1, bs, emb_dim), env_weights: (bs, 1, 1)
        macro_output, _ = self.macro_multihead_attn(query=v_p_mac.unsqueeze(0),
                                                    key=key,
                                                    value=value)
        micro_output, _ = self.micro_multihead_attn(query=v_p_mic.unsqueeze(0),
                                                    key=key,
                                                    value=value)

        output = torch.cat([
            detector_output,
            macro_output.squeeze(0),
            micro_output.squeeze(0)
        ],
                           dim=-1)
        return output

    def forward_gate(self, detector_output=None, v_p_mac=None, v_p_mic=None):
        # (bs, env_dim)
        g = torch.sigmoid(
            self.W_gate(torch.cat([detector_output, v_p_mac], dim=-1)))
        # (bs, env_dim)
        v_p = g * v_p_mac + (1 - g) * v_p_mic
        # (bs, env_dim + FEND_last_output)
        output = torch.cat([detector_output, v_p], dim=-1)
        return output

    def calculate_loss(self, data) -> Tensor:
        """
        calculate loss via CrossEntropyLoss

        Args:
            data (dict): data dict, keys: ['post_simcse', 'avg_mac', 'avg_mic', 'p_mac', 'p_mic', 'avg_mic_mic', 'token', 'label']

        Returns:
            Tensor: loss
        """

        post_simcse, avg_mac, avg_mic, p_mac, p_mic, avg_mic_mic, token, label = data.values(
        )
        output = self(post_simcse,
                      avg_mac,
                      avg_mic,
                      p_mac,
                      p_mic,
                      avg_mic_mic,
                      token=token)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        return loss

    def predict(self, data_without_label):
        """
        predict the probability of being fake

        Args:
            data_without_label (dict): data dict, keys: ['post_simcse', 'avg_mac', 'avg_mic', 'p_mac', 'p_mic', 'avg_mic_mic', 'token']

        Returns:
            Tensor: softmax probability, shape=(batch_size, 2)
        """

        post_simcse, avg_mac, avg_mic, p_mac, p_mic, avg_mic_mic, token, _ = data_without_label.values(
        )
        output = self(post_simcse,
                      avg_mac,
                      avg_mic,
                      p_mac,
                      p_mic,
                      avg_mic_mic,
                      token=token)
        return F.softmax(output, dim=-1)
