import datetime
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
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
        optimizer_initializer = None,
        loss_func: Callable = None,
        metric_funcs= {},
        max_epochs=100,
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

        # check validation is available
        vaildation_available = self.valid_dataloader is not None

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
            for idx, (x, y) in enumerate(self.train_dataloader):
                if self.cuda:
                    x, y = x.cuda(), y.cuda()
                self.optimizer.zero_grad()

                y_hat = self.model(x)
                loss = self.lossfunc(y_hat, y)

                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if vaildation_available:
                y_hats = []
                ys = []
                self.model.eval()
                with torch.no_grad():
                    for idx, (x, y) in enumerate(self.valid_dataloader):
                        if self.cuda:
                            x, y = x.cuda(), y.cuda()
                        y_hat = self.model(x)
                        y_hats.append(y_hat.detach().cpu())
                        ys.append(y.detach().cpu())

                val_y_hats, val_ys = torch.cat(y_hats, dim=0), torch.cat(ys, dim=0)

                metric = self.metrics(val_y_hats, val_ys)

                if best_metric < metric[0][1]:
                    best_metric = metric[0][1]
                    self.save_model(
                        ckpt_dir_path, f"epoch={epoch}_metric={best_metric:.5f}.pt"
                    )

            self._record_loss_metrics(
                epoch, loss.detach().cpu(), metric, writer, history_csv_path
            )

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
            ret.append((name, func(y_hat, y)))
        return ret

    def save_model(self, path, name):
        torch.save(self.model, Path(path) / name)

    def load_model(self):
        return NotImplementedError()

    def _record_loss_metrics(
        self, epoch, loss=None, metrics=None, writer=None, log_csv_path=None
    ):
        now = datetime.datetime.now()
        # write epoch result into csv file
        if log_csv_path is not None:
            log_csv_path = Path(log_csv_path)
            is_new_writing = not os.path.exists(log_csv_path)

            with open(log_csv_path, "a+") as file:
                # write header
                if is_new_writing:
                    header = "epoch,"
                    if loss is not None:
                        header += "loss,"
                    if metrics is not None:
                        # m = (metric_name, score)
                        header += ",".join(list(zip(*metrics))[0])
                    header += ",time"
                    file.write(header + "\n")

                # write row
                row = f"{epoch},"
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
        row = f"Epoch : {epoch} | loss : {loss} |"
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
