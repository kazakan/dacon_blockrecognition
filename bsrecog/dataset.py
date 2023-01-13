import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.io import read_image

STANFORD_BACKGROUND_DATASET_URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"


class StanfordBackgroundDataset(Dataset):
    """
    Download Background image data from work below.

    S. Gould, R. Fulton, D. Koller. Decomposing a Scene into Geometric and Semantically Consistent Regions. Proceedings of International Conference on Computer Vision (ICCV), 2009
    """

    def __init__(self, path: os.PathLike, transform=None):

        self.path = Path(path)
        self.transform = transform

        if not os.path.exists(path):
            os.makedirs(path)

        # download if given path is not empty
        if os.path.isdir(self.path):
            if not os.listdir(self.path):
                print(f"Download files from {STANFORD_BACKGROUND_DATASET_URL}")
                urllib.request.urlretrieve(
                    STANFORD_BACKGROUND_DATASET_URL, "file.tar.gz"
                )
                tar = tarfile.open("file.tar.gz", "r:gz")
                tar.extractall(path=self.path)
                tar.close()
        else:
            print(f"Folder is not empty. Use Files in {self.path/'iccv09Data/images'}.")

        cwd = Path.cwd()
        self.bg_imgs = sorted(
            [
                str(p.resolve().relative_to(cwd))
                for p in self.path.glob("iccv09Data/images/*.jpg")
            ]
        )

    def __getitem__(self, index):
        if self.transform:
            return self.transform(read_image(self.bg_imgs[index])) / 256
        return read_image(self.bg_imgs[index]) / 255

    def __len__(self):
        return len(self.bg_imgs)

    def get_random(self):
        return self[random.randint(0, len(self.bg_imgs) - 1)]


class BlockDataset(Dataset):
    def __init__(
        self,
        directory: Union[str, os.PathLike],
        csv_path: Union[str, os.PathLike] = None,
        transform=None,
    ):
        self.img_dir = Path(directory)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.img_paths = None
        self.transform = transform

        if not os.path.isdir(directory):
            raise Exception("directory is not folder path")

        if csv_path:
            if not os.path.exists(csv_path):
                raise Exception("csv_path is given but file not exists")

            df = pd.read_csv(csv_path)

            self.img_paths = [self.img_dir / (p + ".jpg") for p in df["id"].values]
            self.y = torch.tensor(
                df[["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]].values
            )
        else:
            self.img_paths = sorted(list(self.img_dir.glob("*.jpg")))
            self.y = torch.zeros((len(self.img_paths), 10))  # dummy y

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = read_image(str(self.img_paths[index])) / 256
        if self.transform:
            img = self.transform(img)
        return img, self.y[index, :]


class BlendBackgroundTransform:
    def __init__(self, background_dataset: StanfordBackgroundDataset):
        self.backgrounds = background_dataset

    def __call__(self, img: torch.Tensor):
        # mask area [(white background) | (black background after rotate)]
        mask = img.mean(dim=0).gt(0.8) | (img.mean(dim=0) == 0)
        ret = self.backgrounds.get_random()

        ret[0, :, :] = torch.where(mask, ret[0, :, :], img[0, :, :])
        ret[1, :, :] = torch.where(mask, ret[1, :, :], img[1, :, :])
        ret[2, :, :] = torch.where(mask, ret[2, :, :], img[2, :, :])

        # ret = transforms.GaussianBlur(3,1)(ret)

        return ret


# data loader
def prepare_dataloader(
    train_img_dir_path=None,
    train_csv_path=None,
    test_img_dir_path=None,
    valid_set_ratio=0.2,
    background_path="./",
    img_size=(256, 256),
    batch_size=64,
    debug=False,
):

    train_dataset, valid_dataset, test_dataset = None, None, None

    if (train_img_dir_path is not None) and (train_csv_path is not None):
        background_set = StanfordBackgroundDataset(
            path=background_path, transform=transforms.Resize(img_size)
        )

        original_train_dataset = BlockDataset(
            directory=train_img_dir_path,
            csv_path=train_csv_path,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.RandomRotation(degrees=(-30, 30)),
                    BlendBackgroundTransform(background_set),
                    transforms.GaussianBlur(3, 2),
                ],
            ),
        )

        # if debug mode use only 2 batch
        if debug:
            original_train_dataset = Subset(
                original_train_dataset, list(range(batch_size * 2))
            )

        # split train, valid
        if valid_set_ratio > 0:
            train_dataset, valid_dataset = random_split(
                original_train_dataset, [1 - valid_set_ratio, valid_set_ratio]
            )
        else:
            train_dataset = original_train_dataset
            valid_dataset = None

    # test data
    if test_img_dir_path is not None:
        test_dataset = BlockDataset(test_img_dir_path)
        if debug:
            test_dataset = Subset(test_dataset, list(range(batch_size * 2)))

    train_loader = (
        DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        if train_dataset is not None
        else None
    )
    valid_loader = (
        DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
        if valid_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size)
        if test_dataset is not None
        else None
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # seed_everything()

    bg = StanfordBackgroundDataset("./data/bg")

    data = BlockDataset(
        "./data/train",
        "./data/train.csv",
        transform=transforms.Compose(
            [
                transforms.Resize((255, 255)),
                transforms.RandomRotation(degrees=(-30, 30)),
                BlendBackgroundTransform(bg),
                transforms.GaussianBlur(3, 2),
            ],
        ),
    )

    plt.imshow(torch.permute(data[random.randint(0, 1000)][0], (1, 2, 0)))
    plt.show()
