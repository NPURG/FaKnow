import torch.nn.functional as F
from torch.utils.data import DataLoader

from template.train.trainer import BaseTrainer
from template.utils.pgd import PGD
from template.utils.util import dict2str


class MFANTrainer(BaseTrainer):
    def _train_epoch(self, data: DataLoader, epoch: int):
        self.model.train()

        pgd_word = PGD(self.model, emb_name='word_embedding', epsilon=6, alpha=1.8)
        loss_defence = others = loss_adv = None

        for batch_id, batch_data in enumerate(data):
            # common loss
            loss_defence, others = self.model.calculate_loss(batch_data)
            self.optimizer.zero_grad()
            loss_defence.backward()

            # PGD
            k = 3
            pgd_word.backup_grad()
            for t in range(k):
                pgd_word.attack(is_first_attack=(t == 0))
                if t != k - 1:
                    self.model.zero_grad()
                else:
                    pgd_word.restore_grad()
                y_pred = self.model.predict(batch_data)
                loss_adv = F.cross_entropy(y_pred, batch_data['label'])
                loss_adv.backward()
            pgd_word.restore()

            self.optimizer.step()
        print(f"loss={loss_defence.item()}  {dict2str(others)}  pgd_loss={loss_adv.item()}", end='  ')
