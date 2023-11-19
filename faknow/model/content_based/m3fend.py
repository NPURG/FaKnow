import os
import torch
from torch import Tensor
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import numpy as np
from faknow.model.model import AbstractModel
from faknow.model.layers.layers_m3fend import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from faknow.utils.utils_m3fend import data2gpu, Averager, metrics, Recorder
import logging
import math
from sklearn.cluster import KMeans
import numpy as np
from torch.nn.parameter import Parameter


def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1))


def norm(x):
    length = cal_length(x).view(-1, 1)
    x = x / length
    return x


def convert_to_onehot(label, batch_size, num):
    return torch.zeros(batch_size, num).cuda().scatter_(1, label, 1)


class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        self.domain_memory = dict()

    def forward(self, feature, category):
        feature = norm(feature)
        domain_label = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau,
                                                    dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()

        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

        return domain_att

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim=0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem


class M3FEND(AbstractModel):
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, LNN_dim, domain_num, dataset):
        super(M3FEND, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.domain_num = domain_num
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.LNN_dim = LNN_dim
        print('semantic_num_expert:', self.semantic_num_expert, 'emotion_num_expert:', self.emotion_num_expert,
              'style_num_expert:', self.style_num_expert, 'lnn_dim:', self.LNN_dim)
        self.fea_size = 256
        self.emb_dim = emb_dim
        if dataset == 'ch':
            self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        content_expert = []
        for i in range(self.semantic_num_expert):
            # cnn_extractor 提取出关键的特征信息，即 TextCNN 被用作 SemNet
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        emotion_expert = []
        for i in range(self.emotion_num_expert):
            # MLP 提取出关键的特征信息，即 MLP 被用作 EmoNet
            if dataset == 'ch':
                emotion_expert.append(MLP(47 * 5, [256, 320, ], dropout, output_layer=False))
            elif dataset == 'en':
                emotion_expert.append(MLP(38 * 5, [256, 320, ], dropout, output_layer=False))
        self.emotion_expert = nn.ModuleList(emotion_expert)

        style_expert = []
        for i in range(self.style_num_expert):
            # MLP 提取出关键的特征信息，即 MLP 被用作 StyNet
            if dataset == 'ch':
                style_expert.append(MLP(48, [256, 320, ], dropout, output_layer=False))
            elif dataset == 'en':
                style_expert.append(MLP(32, [256, 320, ], dropout, output_layer=False))
        self.style_expert = nn.ModuleList(style_expert)

        self.gate = nn.Sequential(nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.LNN_dim),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.weight = torch.nn.Parameter(torch.Tensor(self.LNN_dim,
                                                      self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert)).unsqueeze(
            0).cuda()
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if dataset == 'ch':
            self.domain_memory = MemoryNetwork(input_dim=self.emb_dim + 47 * 5 + 48, emb_dim=self.emb_dim + 47 * 5 + 48,
                                               domain_num=self.domain_num, memory_num=self.memory_num)
        elif dataset == 'en':
            self.domain_memory = MemoryNetwork(input_dim=self.emb_dim + 38 * 5 + 32, emb_dim=self.emb_dim + 38 * 5 + 32,
                                               domain_num=self.domain_num, memory_num=self.memory_num)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.all_feature = {}

        self.classifier = MLP(320, mlp_dims, dropout)

    def save_feature(self, **kwargs):
        '''
        这段代码的作用是将所有样本的归一化特征按照它们的领域（domain）信息保存在self.all_feature 字典中。
        键对应一个领域（domain）的整数，值是一个包含该领域所有样本特征的列表。每个特征都以 NumPy 数组的形式表示。
        '''

        # 文本特征
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        # 情感特征
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        # 风格特征
        style_feature = kwargs['style_feature']

        # 类别信息
        category = kwargs['category']

        # 将文本特征、情感特征和风格特征按列拼接成一个总体特征张量 all_feature 并进行归一化
        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)

        # 将所有样本的归一化特征按照它们的领域（domain）信息保存在 self.all_feature 字典中
        # 键对应一个领域（domain）的整数，值是一个包含该领域所有样本特征的列表。每个特征都以 NumPy 数组的形式表示。
        for index in range(all_feature.size(0)):
            domain = int(category[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).cpu().detach().numpy())

    def init_memory(self):
        # 通过 K-Means 聚类，为每个领域创建一个域内存，该域内存包含了该领域内样本特征的聚类中心
        # 这有助于模型学习领域内的代表性特征，提高模型对不同领域数据的适应能力
        for domain in self.all_feature:
            # 数据准备
            all_feature = np.concatenate(self.all_feature[domain])

            # 用 K-Means 聚类算法对 all_feature 进行聚类，将其分成 self.memory_num 个簇
            kmeans = KMeans(n_clusters=self.memory_num, init='k-means++').fit(all_feature)

            # kmeans.cluster_centers_ 包含了每个簇的中心点，即簇心
            #
            centers = kmeans.cluster_centers_

            # 将簇心数组转换为 PyTorch Tensor，并将其存储到域内存中
            # 这个域内存存储了每个领域的聚类中心，用于后续在训练中引入领域信息
            centers = torch.from_numpy(centers).cuda()
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)

    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        category = kwargs['category']

        content_feature = self.bert(content, attention_mask=content_masks)[0]

        gate_input_feature, _ = self.attention(content_feature, content_masks)
        memory_att = self.domain_memory(torch.cat([gate_input_feature, emotion_feature, style_feature], dim=-1),
                                        category)
        domain_emb_all = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda())
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        gate_input = torch.cat([domain_embedding, general_domain_embedding], dim=-1)

        gate_value = self.gate(gate_input).view(content_feature.size(0), 1, self.LNN_dim)

        shared_feature = []
        for i in range(self.semantic_num_expert):
            shared_feature.append(self.content_expert[i](content_feature).unsqueeze(1))

        for i in range(self.emotion_num_expert):
            shared_feature.append(self.emotion_expert[i](emotion_feature).unsqueeze(1))

        for i in range(self.style_num_expert):
            shared_feature.append(self.style_expert[i](style_feature).unsqueeze(1))

        shared_feature = torch.cat(shared_feature, dim=1)

        embed_x_abs = torch.abs(shared_feature)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        embed_x_log = torch.log1p(embed_x_afn)

        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out)
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        shared_feature = torch.bmm(gate_value, shared_feature).squeeze()

        deep_logits = self.classifier(shared_feature)

        return torch.sigmoid(deep_logits.squeeze(1))

    def calculate_loss(self, batch) -> Tensor:
        batch_data = data2gpu(batch, self.use_cuda)
        label = batch_data['label']
        category = batch_data['category']
        label_pred = self.forward(**batch_data)
        loss = self.loss_fn(label_pred, label.float())
        return loss

    def predict(self, data_without_label) -> Tensor:
        # todo


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 dataset,
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.dataset = dataset

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self, logger=None):
        if (logger):
            logger.info('start training......')

        # 模型
        self.model = M3FEND(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
                                 self.style_num, self.lnn_dim, len(self.category_dict), self.dataset)
        if self.use_cuda:
            self.model = self.model.cuda()

        # 损失函数
        loss_fn = torch.nn.BCELoss()

        # 优化器
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 学习率调整器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        # 创建一个用于记录训练过程并执行早停的 Recorder 对象。在模型训练过程中，会使用这个对象来记录和监控模型的性能，并在达到早停条件时停止训练。
        recorder = Recorder(self.early_stop)

        # 设置模型为训练模式
        self.model.train()

        # 创建进度条对象
        train_data_iter = tqdm.tqdm(self.train_loader)

        # 循环遍历训练数据 : 将数据移动到GPU + 将所有样本的归一化特征按照它们的领域（domain）信息保存在self.all_feature 字典中
        for step_n, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, self.use_cuda)
            label_pred = self.model.save_feature(**batch_data)

        # Domain Event Memory 初始化
        self.model.init_memory()
        print('initialization finished')

        # 开始训练
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                label_pred = self.model(**batch_data)

                # 计算损失
                loss = loss_fn(label_pred, label.float())

                # 梯度清零
                optimizer.zero_grad()

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                with torch.no_grad():
                    self.model.write(**batch_data)
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(),
                                                                                   avg_loss.item())
            self.model.train()
            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_m3fend.pkl'))
                self.best_mem = self.model.domain_memory.domain_memory
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_m3fend.pkl')))
        self.model.domain_memory.domain_memory = self.best_mem
        results = self.test(self.test_loader)
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_m3fend.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict)
