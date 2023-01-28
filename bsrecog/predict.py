import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

    if type(args.model) is list:
        models = []
        for path in args.model:
            if os.path.isfile(path):
                models.append(torch.load(path))
            else:
                raise Exception("Wrong model argument.")
    elif os.path.isfile(args.model):
        models = [torch.load(args.model)]
    else:
        raise Exception("Wrong model argument.")

    # check output file path
    with open(args.submission_file_path, "w") as file:
        pass

    # check for tta
    if args.tta < 2:
        args.tta = 1

    _, _, test_dataloader = prepare_dataloader(
        test_img_dir_path=args.img_dir_path,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        tta=args.tta,
        debug=args.debug,
    )

    results = []

    for model in models:
        trainer = Trainer(
            model,
            test_dataloader=test_dataloader,
            seed=args.seed,
            cuda=args.cuda,
        )

        for _ in range(args.tta):
            result = trainer.predict()
            results.append(result)

    results = torch.stack(results)
    result = results.mean(dim=0)
    result = (result >= 0.5).int()

    ids = [x.stem for x in sorted(Path(args.img_dir_path).glob("*.jpg"))]

    with open(args.submission_file_path, "w") as file:
        file.write("id,A,B,C,D,E,F,G,H,I,J\n")
        for id, tensor in zip(ids, result):
            row = f"{id}," + ",".join([str(v.item()) for v in tensor])
            file.write(row + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="+")
    parser.add_argument("img_dir_path")
    parser.add_argument("submission_file_path")

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--img-size", default=256,type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--tta", default=1, type=int)
    parser.add_argument("--cuda", default=False, action="store_true")

    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()
