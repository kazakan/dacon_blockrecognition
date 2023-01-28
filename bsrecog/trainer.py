import datetime
import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bsrecog.utils import now_to_str, seed_everything


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        valid_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        name: str = "experiment_name",
        optimizer_initializer=None,
        loss_func: Callable = None,
        metric_funcs={},
        max_epochs=100,
        valid_interval=None,
        model_ckpt_root_path="./ckpts",
        seed=42,
        cuda=False,
    ):
        self.name = name
        self.seed = seed
        self.max_epoch = max_epochs
        self.cuda = cuda

        self.model_ckpt_root_path = Path(model_ckpt_root_path)
        self.model = model

        self.metric_funcs = metric_funcs

        self.lossfunc = loss_func

        self.optimizer_initializer = optimizer_initializer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.valid_interval = valid_interval

        if self.train_dataloader is None and self.test_dataloader is None:
            raise Exception("train_dataset and test_dataset are both None.")

    def train(self):
        # seed all
        seed_everything(self.seed)
        if self.cuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        # check train data is available
        if self.train_dataloader is None:
            raise Exception("Invalid train_loader. Cannot train now.")

        # init optimizer
        self.optimizer, self.lr_scheduler = self.optimizer_initializer(self.model)

        # check optimizer is available
        if self.optimizer is None:
            raise Exception(
                "self.optimizer is None. Somethingwent wrong in optimizer_initializer"
            )

        # set validation interval
        if self.valid_interval is None:
            self.valid_interval = len(self.train_dataloader)
        elif type(self.valid_interval) is float:
            assert self.valid_interval > 0 and self.valid_interval <= 1
            self.valid_interval = int(len(self.train_dataloader) * self.valid_interval)
            assert self.valid_interval >= 1
        elif type(self.valid_interval) is int:
            assert (self.valid_interval > 0) and (
                self.valid_interval <= len(self.train_dataloader)
            )
        else:
            raise Exception(f"Wrong valid_interval type ({type(self.valid_interval)})")

        # check checkpoint directory path
        ckpt_dir_path = self.model_ckpt_root_path / self.name
        if not os.path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
        if not ckpt_dir_path.is_dir():
            raise Exception(f"Path {ckpt_dir_path} is not directory.")

        history_csv_path = ckpt_dir_path / "history.csv"

        # init tensorboard writer
        writer = SummaryWriter()

        best_metric = float("-inf")

        print(f"Traning started. {now_to_str()}")

        for epoch in range(self.max_epoch):
            self.model.train()
            losses = []
            for idx, (x, y) in enumerate(self.train_dataloader):
                
                self.optimizer.zero_grad()
                # y = y.float()
                
                if self.cuda:
                    x, y = x.cuda(), y.cuda()

                with torch.autocast("cuda" if self.cuda else "cpu"):
                    y_hat = self.model(x)
                    loss = self.lossfunc(y_hat, y)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.detach().cpu())

                if (((idx + 1) % self.valid_interval) == 0) or (
                    (idx + 1) == len(self.train_dataloader)
                ):
                    metric = self._validation(ckpt_dir_path, epoch, idx, best_metric)

                    self._record_loss_metrics(
                        epoch,
                        idx,
                        torch.stack(losses).mean(),
                        metric,
                        writer,
                        history_csv_path,
                    )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def predict(self, dataloader: DataLoader = None, return_y=False):

        if dataloader is None:
            dataloader = self.test_dataloader

        if dataloader is None:
            raise Exception("Invalid dataloader in predict. Give proper dataloader.")

        if self.cuda:
            self.model = self.model.cuda()

        y_hats = []
        ys = []
        self.model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):
                if self.cuda:
                    x, y = x.cuda(), y.cuda()
                y_hat = self.model(x)
                y_hats.append(y_hat.detach().cpu())

                if return_y:
                    ys.append(y)

        if return_y:
            return torch.cat(y_hats, dim=0).detach().cpu(), torch.cat(ys, dim=0).cpu()
        return torch.cat(y_hats, dim=0).detach().cpu()

    def metrics(self, y_hat, y):
        ret = []
        for name, func in self.metric_funcs.items():
            ret.append((name, func(y_hat, y).detach().cpu()))
        return ret

    def save_model(self, path, name):
        torch.save(self.model, Path(path) / name)

    def load_model(self):
        return NotImplementedError()

    def _validation(self, ckpt_dir_path, epoch, step, best_metric):
        if self.valid_dataloader is not None:
            self.model.eval()
            with torch.no_grad():
                y_hats = []
                ys = []

                for idx, (x, y) in enumerate(self.valid_dataloader):
                    y = y.float()
                    if self.cuda:
                        x, y = x.cuda(), y.cuda()
                    y_hat = self.model(x)
                    y_hats.append(y_hat)
                    ys.append(y)

            val_y_hats, val_ys = torch.cat(y_hats, dim=0), torch.cat(ys, dim=0)

            metric = self.metrics(val_y_hats, val_ys)

            if best_metric < metric[0][1]:
                best_metric = metric[0][1]
                self.save_model(
                    ckpt_dir_path,
                    f"epoch={epoch}_step={step}_metric={best_metric:.5f}.pt",
                )
            return metric
        return None

    def _record_loss_metrics(
        self, epoch, step, loss=None, metrics=None, writer=None, log_csv_path=None
    ):
        now = datetime.datetime.now()
        # write epoch result into csv file
        if log_csv_path is not None:
            log_csv_path = Path(log_csv_path)
            is_new_writing = not os.path.exists(log_csv_path)

            with open(log_csv_path, "a+") as file:
                # write header
                if is_new_writing:
                    header = "epoch,step,"
                    if loss is not None:
                        header += "loss,"
                    if metrics is not None:
                        # m = (metric_name, score)
                        header += ",".join(list(zip(*metrics))[0])
                    header += ",time"
                    file.write(header + "\n")

                # write row
                row = f"{epoch},{step},"
                if loss is not None:
                    row += f"{loss},"
                if metrics is not None:
                    # m = (metric_name, score)
                    row += ",".join([str(v.item()) for v in list(zip(*metrics))[1]])
                row += f",{str(now)}"
                file.write(row + "\n")

        # write epoch result into Tensorboard
        if writer is not None:
            for m in metrics:
                writer.add_scalar(m[0], m[1], epoch)

        # print
        row = f"Epoch : {epoch} | Step : {step} | loss : {loss : .8f} |"
        for m in metrics:
            row += f" {m[0]} : {m[1] : .6f} |"
        row += f" {now_to_str()} |"
        print(row)


# metrics
def acc(y_hat: torch.Tensor, y: torch.Tensor):
    return (y == (y_hat >= 0.5)).sum() / torch.numel(y)


def tp(y_hat: torch.Tensor, y: torch.Tensor):
    return ((y_hat >= 0.5) and (y == 1)).sum()


def precision(y_hat: torch.Tensor, y: torch.Tensor):
    return tp(y_hat, y) / (y_hat == 1).sum()


def recall(y_hat: torch.Tensor, y: torch.Tensor):
    return tp(y_hat, y) / (y == 1).sum()
