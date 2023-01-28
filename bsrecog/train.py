import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.optim as optim

from bsrecog.dataset import prepare_dataloader
from bsrecog.model import *
from bsrecog.trainer import Trainer, acc
from bsrecog.utils import now_to_str, seed_everything


def train(args):
    seed_everything(args.seed)

    # init ckpt folder
    experiment_name = args.name + "_" + now_to_str()
    ckpt_dir_path = Path(args.ckpt_root_path) / experiment_name
    if not os.path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)
    if not ckpt_dir_path.is_dir():
        raise Exception(f"Path {ckpt_dir_path} is not directory.")

    # if debug mode
    if args.debug:
        args.batch_size = 10
        args.max_epochs = 3

    # record args into json
    with open(ckpt_dir_path / "args.json", "w") as argfile:
        json.dump(vars(args), argfile, indent="\t")

    if args.model in MODEL_LIST:
        if args.model == "EfficientNetB0_Model":
            model = EfficientNetB0_Model(use_pretrained=args.use_pretrained)
        elif args.model == "EfficientNetV2_M_Model":
            model = EfficientNetV2_M_Model(use_pretrained=args.use_pretrained)
        elif args.model == "EfficientNetV2_L_Model":
            model = EfficientNetV2_L_Model(use_pretrained=args.use_pretrained)
        elif args.model == "ResNet50_Model":
            model = ResNet50_Model(use_pretrained=args.use_pretrained)
        elif args.model == "ConvNeXt_Base_Model":
            model = ConvNeXt_Base_Model(use_pretrained=args.use_pretrained)
        elif args.model == "ConvNeXt_Small_Model":
            model = ConvNeXt_Small_Model(use_pretrained=args.use_pretrained)
        elif args.model == "PR_EfficientNetV2_M_Model":
            model = PR_EfficientNetV2_M_Model(use_pretrained=args.use_pretrained)
    elif os.path.isfile(args.model):
        model = torch.load(args.model)
    else:
        raise Exception("Wrong model argument.")

    train_dataloader, valid_dataloader, _ = prepare_dataloader(
        train_img_dir_path=args.img_dir_path,
        train_csv_path=args.csv_path,
        test_img_dir_path=None,
        valid_set_ratio=args.valid_set_ratio,
        n_splits=args.n_splits,
        current_fold=args.cur_fold,
        background_path=args.background_path,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        debug=args.debug,
    )

    def init_optimizer(_model: nn.Module):
        optimizer = optim.AdamW(_model.parameters(), args.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.t_max, eta_min=args.eta_min
        )
        return optimizer, lr_scheduler

    trainer = Trainer(
        model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        valid_interval=args.validation_interval,
        optimizer_initializer=init_optimizer,
        loss_func=nn.BCELoss(),
        metric_funcs={"ACC": acc},
        max_epochs=args.max_epochs,
        model_ckpt_root_path=args.ckpt_root_path,
        seed=args.seed,
        cuda=args.cuda,
        name=experiment_name,
    )

    trainer.train()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model class name or model path")
    parser.add_argument("img_dir_path", help="Path to train image folder")
    parser.add_argument("csv_path", help="Path to train csv file")
    parser.add_argument(
        "--background_path",
        default="./data/bg",
        help="Path to save background data. If given path is not empty, program regards background data exists.",
    )
    parser.add_argument(
        "--ckpt-root-path", default="./ckpts", help="Path to save checkpoint folder"
    )

    parser.add_argument(
        "--use-pretrained",
        default=False,
        action="store_true",
        help="Choose to use pretrained weight using imagenet.",
    )

    parser.add_argument(
        "--valid-set-ratio",
        default=0.2,
        type=float,
        help="Ratio of (validset size) / (original train data). If --n-splits is not 0, --valid-set-ratio is ignored.",
    )
    parser.add_argument(
        "--validation-interval", default=None, type=float, help="Interval of validation"
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--img-size", default=256,type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument(
        "--t-max", default=50, type=int, help="T_max in CosineAnnealingLr"
    )
    parser.add_argument(
        "--eta-min", default=1e-6, type=float, help="eta_min in CosineAnnealingLr"
    )
    parser.add_argument("--max-epochs", default=100, type=int)

    parser.add_argument(
        "--n-splits",
        default=0,
        type=int,
        help="Number of folds to split. If 0, split using --valid-set-ratio.",
    )
    parser.add_argument(
        "--cur-fold",
        default=0,
        type=int,
        help="Current fold number to use. If --n-splits is 0, ignored.",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--name", default="EXPERIMENT", type=str, help="Name of experiment"
    )

    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="Use cuda or not",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode. In Debug mode, only small set of data is used to test whether program totally runs well.",
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
