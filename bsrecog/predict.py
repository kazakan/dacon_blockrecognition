import os
from pathlib import Path

import torch

from bsrecog.dataset import prepare_dataloader
from bsrecog.trainer import Trainer
from bsrecog.utils import seed_everything


def predict(args):
    seed_everything(42)

    # if debug mode
    if args.debug:
        args.batch_size = 10
        args.max_epochs = 3

    if os.path.isfile(args.model):
        model = torch.load(args.model)
    else:
        raise Exception("Wrong model argument.")

    # check output file path
    with open(args.submission_file_path, "w") as file:
        pass

    _, _, test_dataloader = prepare_dataloader(
        test_img_dir_path=args.img_dir_path,
        img_size=(256, 256),
        batch_size=args.batch_size,
        debug=args.debug,
    )

    trainer = Trainer(
        model,
        test_dataloader=test_dataloader,
        seed=args.seed,
        cuda=args.cuda,
    )

    # post processing
    ids = [x.stem for x in sorted(Path(args.img_dir_path).glob("*.jpg"))]
    result = trainer.predict()
    result = (result >= 0.5).int()

    with open(args.submission_file_path, "w") as file:
        file.write("id,A,B,C,D,E,F,G,H,I,J\n")
        for id, tensor in zip(ids, result):
            row = f"{id}," + ",".join([str(v.item()) for v in tensor])
            file.write(row + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("img_dir_path")
    parser.add_argument("submission_file_path")

    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", default=False, action="store_true")

    parser.add_argument("--debug", default=False, type=bool, action="store_true")

    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()
