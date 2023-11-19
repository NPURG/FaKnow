import os

import torch
from tqdm import tqdm

from faknow.model.content_based.m3fend import M3FEND
from faknow.utils.utils_m3fend import Recorder, data2gpu, Averager, metrics


class M3FENDTrainer():
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
                 epochs=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epochs = epochs
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
        for epoch in range(self.epochs):
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
