import torch
from torch.utils.data.dataloader import DataLoader

from template.train.trainer import BaseTrainer


class EANNTrainer(BaseTrainer):

    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        self.model.eval()
        dataloader = DataLoader(data, batch_size, shuffle=True)
        outputs = []
        labels = []
        for text, image, label, other_data in dataloader:
            # todo 修改Trainer
            #  对于多个输出的模型，传入要使用的输出的下标index即可
            # outputs.append(self.model(text, image, other_data['mask'])[index])
            outputs.append(self.model(text, image, other_data['mask'])[0])
            labels.append(label)
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(labels))

    def _train_epoch(self, data, batch_size: int, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        dataloader = DataLoader(data, batch_size, shuffle=True)

        p = float(epoch) / 100
        lr = 0.001 / (1. + 10 * p) ** 0.75
        self.optimizer.lr = lr

        for batch_id, (text, image,
                       label, other_data) in enumerate(dataloader):
            # text, image, train_mask, train_labels, event_labels = \
            #     train_data[0], train_data[1], train_data[2], \
            #     train_labels, event_labels

            # todo 不要重写trainer 方法，只需在model中写calculate loss函数
            # 然后再此处调用即可，参考recbole
            class_outputs, domain_outputs = self.model(text, image,
                                                       other_data['mask'])

            label = label.long()
            class_loss = self.criterion(class_outputs, label)
            event_labels = other_data['event_label'].long()
            domain_loss = self.criterion(domain_outputs, event_labels)
            loss = class_loss + domain_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f'class_loss={class_loss}, domain_loss={domain_loss}')
        return loss
