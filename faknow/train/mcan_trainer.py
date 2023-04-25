import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from faknow.train.trainer import BaseTrainer
from faknow.utils.util import dict2str


class MCANTrainer(BaseTrainer):
    def _train_epoch(
            self,
            loader: DataLoader,
            epoch: int,
            writer: SummaryWriter
    ):
        self.model.train()

        pbar = tqdm(enumerate(loader), total=len(loader), ncols=100, desc='Training')

        loss = others = None
        for batch_id, batch_data in pbar:
            result = self.model.calculate_loss(batch_data)

            if type(result) is tuple and len(result) == 2 and type(result[1]) is dict:
                loss = result[0]
                others = result[1]
            elif type(result) is torch.Tensor:
                loss = result
            else:
                raise TypeError(f"result type error: {type(result)}")

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=1.0)
            self.optimizer.step()

            pbar.set_postfix_str(f"loss={loss.item()}")

        pbar.close()

        writer.add_scalar("Train/loss", loss.item(), epoch)
        if others is None:
            writer.add_scalar("Train/loss", loss.item(), epoch)
            self.logger.info(f"training loss : loss={loss.item():.8f}")
            tqdm.write(f"training loss : loss={loss.item():.8f}")
        else:
            for metric, value in others.items():
                writer.add_scalar("Train/" + metric, value, epoch)
            self.logger.info(f"training loss : loss={loss.item():.8f}    " + dict2str(others))
            tqdm.write(f"training loss : loss={loss.item():.8f}    " + dict2str(others))