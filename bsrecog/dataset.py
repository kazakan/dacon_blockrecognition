import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset, random_split

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
        img = cv2.imread(str(self.bg_imgs[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 256

        if self.transform is not None:
            return self.transform(image=img)["image"]
        return img

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
        img = cv2.imread(str(self.img_paths[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 256

        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.y[index, :]


class BlendBackgroundTransform(A.ImageOnlyTransform):
    def __init__(
        self, background_dataset: StanfordBackgroundDataset, always_apply=False, p=0.5
    ):
        super(BlendBackgroundTransform, self).__init__(always_apply=always_apply, p=p)
        self.backgrounds = background_dataset

    def apply(self, image, **arg):
        # mask area [(white background) | (black background after rotate)]
        mask = image.mean(axis=2) > 0.8  # | (image.mean(axis=2) < 0.05)
        ret = self.backgrounds.get_random()

        ret[:, :, 0] = np.where(mask, ret[..., 0], image[..., 0])
        ret[:, :, 1] = np.where(mask, ret[..., 1], image[..., 1])
        ret[:, :, 2] = np.where(mask, ret[..., 2], image[..., 2])

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
    tta=0,
    debug=False,
):

    train_dataset, valid_dataset, test_dataset = None, None, None

    if (train_img_dir_path is not None) and (train_csv_path is not None):
        background_set = StanfordBackgroundDataset(
            path=background_path,
            transform=A.Compose(
                [
                    A.Resize(img_size[0], img_size[1]),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomBrightnessContrast(),
                    A.RandomResizedCrop(img_size[0], img_size[1]),
                ]
            ),
        )

        original_train_dataset = BlockDataset(
            directory=train_img_dir_path,
            csv_path=train_csv_path,
            transform=A.Compose(
                [
                    A.Resize(img_size[0], img_size[1]),
                    A.RandomResizedCrop(
                        img_size[0], img_size[1], scale=(0.7, 1), ratio=(0.95, 1.05)
                    ),
                    BlendBackgroundTransform(background_set, p=0.7),
                    A.MedianBlur(3, always_apply=True, p=1),
                    A.HorizontalFlip(),
                    A.RandomBrightnessContrast(),
                    A.Rotate((-30, 30)),
                    ToTensorV2(),
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
        test_aug = A.Compose(
            [
                A.Resize(img_size[0], img_size[1]),
                A.MedianBlur(3, always_apply=True, p=1),
                A.RandomBrightnessContrast(),
                A.Rotate((-30, 30)),
                A.RandomResizedCrop(
                    img_size[0], img_size[1], scale=(0.7, 1), ratio=(0.95, 1.05)
                ),
                A.HorizontalFlip(),
                ToTensorV2(),
            ]
        )
        test_dataset = BlockDataset(
            test_img_dir_path, transform=None if tta < 2 else test_aug
        )
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

    sz = 256

    bg = StanfordBackgroundDataset(
        "./data/bg",
        transform=A.Compose(
            [
                A.Resize(sz, sz),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomBrightnessContrast(),
                A.RandomResizedCrop(sz, sz),
            ]
        ),
    )

    data = BlockDataset(
        "./data/train",
        "./data/train.csv",
        transform=A.Compose(
            [
                A.Resize(sz, sz),
                A.RandomResizedCrop(sz, sz, scale=(0.7, 1), ratio=(0.95, 1.05)),
                BlendBackgroundTransform(bg),
                A.MedianBlur(3, always_apply=True, p=1),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.Rotate((-30, 30)),
                ToTensorV2(),
            ],
        ),
    )

    plt.imshow(torch.permute(data[random.randint(0, 1000)][0], (1, 2, 0)))
    plt.show()
