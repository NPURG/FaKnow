from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BertModel

from template.model.model import AbstractModel

"""
Towards Fine-Grained Reasoning for Fake News Detection
paper: https://arxiv.org/abs/2110.15064
code: https://github.com/Ahren09/FinerFact
"""


def kernel_mus(n_kernels):
    """
    get the mean mus for each gaussian kernel_num. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mus.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mus: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels, sigma_val):
    assert n_kernels >= 1
    l_sigma = [0.001] + [sigma_val] * (n_kernels - 1)
    return l_sigma


class _BertExtractor(nn.Module):
    def __init__(self, bert_name: str):
        super(_BertExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        return last_hidden_state, pooled_output


class FinerFact(AbstractModel):
    """
    Towards Fine-Grained Reasoning for Fake News Detection
    """

    def __init__(self,
                 bert_name: str,
                 evidence_num=5,
                 sigma=0.1,
                 kernel_num=11,
                 dropout=0.6,
                 mode='FF',
                 mean=0.0,
                 std=0.001,
                 channel=6,
                 tweet_num=6,
                 user_num=32,
                 word_num=7,
                 user_embed_dim=64,
                 user_embedding: Optional[Tensor] = None,
                 user_field_num=8):
        """

        Args:
            bert_name (str): name or local path of the bert model
            evidence_num (int): number of evidences. Default=5
            sigma (float): sigma value for gaussian kernel. Default=0.1
            kernel_num (int): number of kernels. Default=11
            dropout (float): dropout rate. Default=0.6
            mode (str): 'FF' or 'FF+'. Default=FF
            mean (float): mean for normal initial weights. Default=0.0
            std (float): std for normal initial weights. Default=0.001
            channel (int): number of channels in each pair. Default=6
            tweet_num (int): number of tweets in each pair. Default=6
            user_num (int): number of users in each pair. Default=32
            word_num (int): number of words in each tweet. Default=7
            user_embed_dim (int): dimension of user embedding. Default=64
            user_embedding (Tensor): pretrained user embedding. If None, user embedding will be generated from scratch. Default=None
            user_field_num (int): number of user fields for training user embedding from scratch. Default=8
        """
        super(FinerFact, self).__init__()

        self.bert_extractor = _BertExtractor(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.evidence_num = evidence_num
        self.kernel_num = kernel_num
        self.sigma_value = sigma
        self.mode = mode

        # projection layers
        self.proj_inference_de = nn.Linear(
            self.bert_extractor.bert_hidden_size * 2, 2)
        self.proj_att = nn.Linear(self.kernel_num, 1)
        self.proj_input_de = nn.Linear(self.bert_extractor.bert_hidden_size,
                                       self.bert_extractor.bert_hidden_size)
        self.proj_select = nn.Linear(self.kernel_num, 1)

        # gaussian kernels
        self.mus = torch.FloatTensor(kernel_mus(self.kernel_num)).view(
            1, 1, 1, self.kernel_num)
        self.sigmas = torch.FloatTensor(
            kernel_sigmas(self.kernel_num,
                          self.sigma_value)).view(1, 1, 1, self.kernel_num)

        # projection layers
        self.proj_pred_P = nn.Linear(channel, 1)
        self.att_prior_P = nn.Linear(tweet_num, 1, bias=False)
        self.att_prior_U = nn.Linear(user_num, 1, bias=False)
        self.att_prior_K = nn.Linear(word_num, 1, bias=False)
        self.proj_pred_interact = nn.Linear(2, 1)

        # user features
        self.user_embed_dim = user_embed_dim
        self.user_num = user_num

        # whether to use pretrained user embedding
        if user_embedding is None:
            self.user_embedding = nn.Linear(user_field_num,
                                            self.user_embed_dim,
                                            bias=False)
            nn.init.normal_(self.user_embedding.weight, mean, std)
            self.pretrain_user = False
        else:
            self.pretrain_user = True
            self.user_embedding = nn.Embedding.from_pretrained(
                user_embedding.reshape(-1, self.user_embed_dim))

        # gat
        self.proj_gat = nn.Sequential(
            nn.Linear(self.bert_extractor.bert_hidden_size * 2, 128),
            nn.ReLU(True), nn.Linear(128, 1))
        if self.mode == "FF-concat":
            """
            This part has been changed for concatenating
            Ignore this part for now
            """
            self.proj_gat_user = nn.Sequential(
                nn.Linear(self.bert_extractor.bert_hidden_size * 2 +
                          self.user_embed_dim,
                          128,
                          bias=False), nn.ReLU(True),
                nn.Linear(128, 1, bias=False))
        else:
            self.proj_gat_user = nn.Sequential(
                nn.Linear(self.user_embed_dim, 128, bias=False), nn.ReLU(True),
                nn.Linear(128, 1, bias=False))
            self.proj_user = nn.Linear(self.user_embed_dim * 2,
                                       self.bert_extractor.bert_hidden_size *
                                       2,
                                       bias=False)
            nn.init.normal_(self.proj_user.weight, mean, std)

        self.step = -1

        self.__init_weights__(mean, std)

    def __init_weights__(self, mean, std):
        nn.init.normal_(self.att_prior_P.weight, mean, std)
        nn.init.normal_(self.att_prior_U.weight, mean, std)
        nn.init.normal_(self.att_prior_K.weight, mean, std)
        nn.init.normal_(self.proj_gat_user[0].weight, mean, std)
        nn.init.normal_(self.proj_gat_user[2].weight, mean, std)

    def get_user_embed(self, user_metadata, batch_size):
        # todo
        # 使用预训练时，是按batch索引的下标
        # 在训练时传入的是user_embeds_train，而在测试时传入的是user_embeds_test
        # 在训练和测试过程中需要切换
        if self.pretrain_user:
            # select pre-trained user embedding for the current batch
            user_num_batch = self.evidence_num * self.user_num * batch_size
            indices = torch.arange(self.step * user_num_batch,
                                   (self.step + 1) * user_num_batch, 1)
            user_embed = self.user_embedding(indices)
            user_embed = user_embed.reshape(-1, self.evidence_num,
                                            self.user_num, self.user_embed_dim)
        else:
            # generate user embedding from user metadata
            user_embed = self.user_embedding(user_metadata)

        return user_embed

    def att_prior_mr(self, R_p, R_u, R_k):
        # saliency
        H_p = self.att_prior_P(R_p)  # (B, 5, 6)
        H_u = self.att_prior_U(R_u)  # (B, 5, 32)
        H_k = self.att_prior_K(R_k)  # (B, 5, 7)
        delta = H_p + H_u + H_k
        return delta

    def self_attention_user(self,
                            inputs,
                            inputs_hiddens,
                            mask,
                            index,
                            z_qv_z_v_all=None):
        """
        Models interactions among user embeddings
        """

        idx = torch.LongTensor([index])
        mask = mask.view([-1, self.evidence_num, self.user_num])

        # Hidden feature of ONLY current node
        # B, num_user, user_embed_dim
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)
        # B, 1, num_user
        own_mask = torch.index_select(mask, 1, idx)

        # B, 1, user_embed_dim
        # x^v of current (one) user
        own_input = torch.index_select(inputs, 1, idx)

        # B, 5, user_embed_dim
        own_input = own_input.repeat(1, self.evidence_num, 1)

        # Hidden feature of ONLY current node
        # B, 5, num_user, user_embed_dim
        own_hidden = own_hidden.repeat(1, self.evidence_num, 1, 1)

        # B, 5, num_user
        own_mask = own_mask.repeat(1, self.evidence_num, 1)

        # B*5, num_user, user_embed_dim
        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)

        # B*5, num_user, user_embed_dim
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        # B*5, num_user
        # gamma: Importance of each user

        att_score = self.get_intersect_matrix_att(
            hiddens_norm.view(-1, self.user_num, self.user_embed_dim),
            own_norm.view(-1, self.user_num, self.user_embed_dim),
            own_mask.view(-1, self.user_num), own_mask.view(-1, self.user_num))

        # B, 5, num_user
        # gamma: Importance of each user
        att_score = att_score.view(-1, self.evidence_num, self.user_num, 1)

        # B, 5, user_embed_dim
        # Token-wise weighted average
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)

        if self.mode == "FF-concat":
            z_qv_z_v = z_qv_z_v_all[:, index, :, :]
            concat_att_embed = torch.cat([z_qv_z_v, denoise_inputs], dim=1)
        else:
            concat_att_embed = denoise_inputs

        # weight_de = torch.cat([own_input, denoise_inputs], -1)
        weight_de = self.proj_gat_user(concat_att_embed)
        weight_de = F.softmax(weight_de, dim=1)
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs_de

    def self_attention(self,
                       inputs,
                       inputs_hiddens,
                       mask,
                       mask_evidence,
                       index,
                       trans_mat_prior=None):
        idx = torch.LongTensor([index])
        max_len = mask.shape[-1]

        # bert_pooler_out: B, 5, 768
        # inputs_hiddens: B, 5, 130, 768

        # B, 5, 130
        mask = mask.view([-1, self.evidence_num, max_len])
        # B, 5, 130
        mask_evidence = mask_evidence.view([-1, self.evidence_num, max_len])
        # B, 130, 768
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)

        # B, 1, 130
        own_mask = torch.index_select(mask, 1, idx)

        # B, 1, 768
        # z^v of current (one) claim-evi pair
        own_input = torch.index_select(inputs, 1, idx)
        # B, 5, 130, 768
        own_hidden = own_hidden.repeat(1, self.evidence_num, 1, 1)

        # B, 5, 130
        own_mask = own_mask.repeat(1, self.evidence_num, 1)

        # B, 5, 768
        own_input = own_input.repeat(1, self.evidence_num, 1)

        # B*5, 130
        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)

        # B, 5, 130, 768
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        # B*5, 130
        # alpha: Importance of each token
        att_score = self.get_intersect_matrix_att(
            hiddens_norm.view(-1, max_len,
                              self.bert_extractor.bert_hidden_size),
            own_norm.view(-1, max_len, self.bert_extractor.bert_hidden_size),
            mask_evidence.view(-1, max_len), own_mask.view(-1, max_len))

        # B, 5, 130, 1
        # alpha: Token-wise weighted average
        att_score = att_score.view(-1, self.evidence_num, max_len, 1)

        # B, 5, 768

        # z^{q -> p}
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)

        # B, 5, 1536
        # z^q || z^{p}
        weight_inp = torch.cat([own_input, inputs], -1)

        # B, 5, 1536
        z_q_z_v = weight_inp

        # MLP()
        # B, 5, 1
        weight_inp = self.proj_gat(weight_inp)

        # gamma
        # B, 5, 1
        weight_inp = F.softmax(weight_inp, dim=1)

        outputs = (inputs * weight_inp).sum(dim=1)

        # B, 5, 1536
        # z^p || z^{q -> p}
        # Can be changed into z^p || z^{q}
        if self.mode == "FF-P":
            weight_de = z_q_z_v  # shallow copy
            denoise_inputs = inputs.clone()
        else:
            weight_de = torch.cat([own_input, denoise_inputs], -1)

        z_qv_z_v = weight_de

        # gamma
        weight_de = self.proj_gat(weight_de)
        if trans_mat_prior is not None:
            weight_de = torch.cat([
                weight_de, trans_mat_prior[:, index].reshape(
                    -1, self.evidence_num, 1)
            ],
                dim=2)
            weight_de = self.proj_pred_interact(weight_de)
        weight_de = F.softmax(weight_de, dim=1)

        # \sum {gamma^{q->p} * \hat{z}^{q->p}}
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs, outputs_de, z_qv_z_v

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1], 1)
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)

        sim = torch.bmm(q_embed,
                        torch.transpose(d_embed, 1,
                                        2)).view(q_embed.size()[0],
                                                 q_embed.size()[1],
                                                 d_embed.size()[1], 1)
        pooling_value = torch.exp(
            (-((sim - self.mus) ** 2) / (self.sigmas ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(
            pooling_value,
            2)  # If merge content and social representation here
        log_pooling_sum = torch.log(torch.clamp(pooling_sum,
                                                min=1e-10)) * attn_q

        log_pooling_sum_all = torch.sum(log_pooling_sum,
                                        1) / (torch.sum(attn_q, 1) + 1e-10)

        log_pooling_sum = self.proj_select(log_pooling_sum_all).view([-1, 1])
        return log_pooling_sum, log_pooling_sum_all

    def get_intersect_matrix_att(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1])
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed,
                        torch.transpose(d_embed, 1,
                                        2)).view(q_embed.size()[0],
                                                 q_embed.size()[1],
                                                 d_embed.size()[1], 1)

        # B, 130, 11
        pooling_value = torch.exp(
            (-((sim - self.mus) ** 2) / (self.sigmas ** 2) / 2)) * attn_d

        # B*5, user_num, 11
        log_pooling_sum = torch.sum(pooling_value, 2)

        # B*5, user_num, 11
        log_pooling_sum = torch.log(torch.clamp(log_pooling_sum, min=1e-10))

        # B*5, user_num, 1
        log_pooling_sum = self.proj_att(log_pooling_sum).squeeze(-1)

        log_pooling_sum = log_pooling_sum.masked_fill_((1 - attn_q).bool(),
                                                       -1e4)
        log_pooling_sum = F.softmax(log_pooling_sum, dim=1)
        return log_pooling_sum

    def predict_prior(self, score):
        prior = self.proj_score(score)
        return prior

    def reshape_input_and_masks(self, bert_hiddens, mask, type_id):
        max_len = mask.shape[-1]

        # B*5, 130
        mask_text = mask.view(-1, max_len).float()

        # First token ([CLS]) set to 0
        # B*5, 1
        mask_text[:, 0] = 0.0

        # Claim part set to 1 (Except first token [CLS])
        mask_claim = (1 - type_id.float()) * mask_text

        # Evidence part set to 1
        mask_evidence = type_id.float() * mask_text

        # z^p or h_p^0, Hidden representation of first token
        inputs_hiddens = bert_hiddens.view(
            -1, max_len, self.bert_extractor.bert_hidden_size)
        inputs_hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=2)

        return mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm

    def channel_text(self, bert_hiddens, bert_pool_out, mask, type_id, delta):
        max_len = mask.shape[-1]

        # mask_text: Every sentence starts with 0, followed by
        # 1's at positions where tokens are present
        mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm = self.reshape_input_and_masks(
            bert_hiddens, mask, type_id)

        # Evidence Selection P(K^v, G)
        # log_pooling_sum_all: Content signals at several kernels
        log_pooling_sum, log_pooling_sum_all = self.get_intersect_matrix(
            inputs_hiddens_norm, inputs_hiddens_norm, mask_claim,
            mask_evidence)
        log_pooling_sum = log_pooling_sum.view([-1, self.evidence_num, 1])
        if not self.mode == "FF-I":
            log_pooling_sum += delta
        select_prob = F.softmax(log_pooling_sum, dim=1)

        # Claim Label Prediction: P(y|K^p, G)
        inputs = bert_pool_out.view(
            [-1, self.evidence_num, self.bert_extractor.bert_hidden_size])
        inputs_hiddens = inputs_hiddens.view([
            -1, self.evidence_num, max_len,
            self.bert_extractor.bert_hidden_size
        ])

        inputs_att_de = []
        z_qv_z_v_all = []

        for i in range(self.evidence_num):
            outputs, outputs_de, z_qv_z_v = self.self_attention(
                inputs, inputs_hiddens, mask_text, mask_text, i)
            inputs_att_de.append(outputs_de)
            z_qv_z_v_all.append(z_qv_z_v)

        # All z^{v}, same as `bert_pooler_out`
        inputs_att = inputs.view(
            [-1, self.evidence_num, self.bert_extractor.bert_hidden_size])
        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        z_qv_z_v_all = torch.cat(z_qv_z_v_all, dim=1)

        # All z^{q->v}
        inputs_att_de = inputs_att_de.view(
            [-1, self.evidence_num, self.bert_extractor.bert_hidden_size])
        z_qv_z_v_all = z_qv_z_v_all.view([
            -1, self.evidence_num, self.evidence_num,
            self.bert_extractor.bert_hidden_size
        ])

        return select_prob, inputs_att, inputs_att_de, z_qv_z_v_all

    def channel_user(self, user_hiddens, z_qv_z_v_all=None):

        # ------------------------------------------
        # Evidence Selection: P(\hat{K}^p, G)
        # ------------------------------------------

        user_hiddens = user_hiddens.reshape(-1, self.evidence_num,
                                            self.user_num, self.user_embed_dim)

        mask_usr = torch.ones_like(user_hiddens[:, :, :, 0])

        inputs_att_de_usr = []

        user_inputs = torch.mean(user_hiddens, dim=2)

        for i in range(self.evidence_num):
            outputs_de_usr = self.self_attention_user(user_inputs,
                                                      user_hiddens, mask_usr,
                                                      i, z_qv_z_v_all)
            inputs_att_de_usr.append(outputs_de_usr)

        # All x^{v}, same as `bert_pooler_out`
        inputs_att_usr = user_inputs.view(
            [-1, self.evidence_num, self.user_embed_dim])
        inputs_att_de_usr = torch.cat(inputs_att_de_usr, dim=1)

        inputs_att_de_usr = inputs_att_de_usr.view(
            [-1, self.evidence_num, self.user_embed_dim])

        return inputs_att_usr, inputs_att_de_usr

    def forward(self,
                token_id,
                mask,
                type_id,
                R_p,
                R_u,
                R_k,
                user_metadata=None):
        """

        Args:
            token_id (Tensor): shape=(batch_size, evidence_num, max_len)
            mask (Tensor): shape=(batch_size, evidence_num, max_len)
            type_id (Tensor): shape=(batch_size, evidence_num, max_len)
            R_p (Tensor): ranking of posts, shape=(batch_size, evidence_num, tweet_num)
            R_u (Tensor): ranking of users, shape=(batch_size, evidence_num, user_num)
            R_k (Tensor): ranking of keywords, shape=(batch_size, evidence_num, word_num)
            user_metadata (Tensor): shape=(batch_size, evidence_num, user_num, user_embed_dim). Default=None

        Returns:
            logits (Tensor): predictions of being fake, shape=(batch_size, 2)
        """
        if (not self.pretrain_user) and user_metadata is None:
            raise ValueError(
                "User metadata is required when pretrained user embedding is not provided."
            )
        if self.pretrain_user and user_metadata is not None:
            print(
                "Warning: User metadata won't be used when pretrained user embedding is provided."
            )

        self.step = self.step + 1
        max_len, batch_size = token_id.shape[-1], token_id.shape[0]

        # reshape
        token_id = token_id.view(-1, max_len)
        mask = mask.view(-1, max_len)
        type_id = type_id.view(-1, max_len)

        bert_hidden_states, bert_pool_out = self.bert_extractor(
            token_id, mask, type_id)
        delta = self.att_prior_mr(R_p, R_u, R_k)

        select_prob, inputs_att, inputs_att_de, z_qv_z_v_all = self.channel_text(
            bert_hidden_states, bert_pool_out, mask, type_id, delta)
        # All z^{q->v} || All z^{v}
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)

        # user features
        user_embed = self.get_user_embed(user_metadata, batch_size)

        if self.mode in ["FF"]:
            # Initialize our embedding module from the embedding matrix
            inputs_att_usr, inputs_att_de_usr = self.channel_user(
                user_embed, z_qv_z_v_all)
            inputs_att_usr_combined = torch.cat(
                [inputs_att_usr, inputs_att_de_usr], -1)
            inputs_att_usr_combined = self.proj_user(inputs_att_usr_combined)

            assert inputs_att_usr_combined.shape == inputs_att.shape
            inputs_att = inputs_att + inputs_att_usr_combined

        inference_feature = self.proj_inference_de(inputs_att)
        class_prob = F.softmax(inference_feature, dim=2)

        if self.mode == "FF-I":
            select_prob = torch.ones_like(select_prob) / self.evidence_num

        prob = torch.sum(select_prob * class_prob, 1)
        logits = torch.log(prob)

        return logits

    def calculate_loss(self, data):
        # token_id, mask, type_id, label, R_p, R_u, R_k, user_metadata = data
        """get item from batch dict"""
        token_id = data['token_id']
        mask = data['mask']
        type_id = data['type_id']
        label = data['label']
        R_p = data['R_p']
        R_u = data['R_u']
        R_k = data['R_k']
        user_metadata = data.get('user_metadata', None)
        logits = self.forward(token_id, mask, type_id, R_p, R_u, R_k,
                              user_metadata)
        loss = F.nll_loss(logits, label)
        return loss

    def predict(self, data):
        # token_id, mask, type_id, R_p, R_u, R_k, user_metadata = data_without_label
        token_id = data['token_id']
        mask = data['mask']
        type_id = data['type_id']
        R_p = data['R_p']
        R_u = data['R_u']
        R_k = data['R_k']
        user_metadata = data.get('user_metadata', None)
        logits = self.forward(token_id, mask, type_id, R_p, R_u, R_k,
                              user_metadata)
        return logits
