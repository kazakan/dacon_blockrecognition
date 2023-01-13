import os

import torch
import torch.nn as nn
from torchvision.models import (ConvNeXt_Base_Weights,
                                EfficientNet_V2_M_Weights, ResNet50_Weights,
                                convnext_base, efficientnet_v2_m, resnet50)

MODEL_LIST = [
    "EfficientNetV2_M_Model",
    "ResNet50_Model",
    "ConvNeXt_Base_Model",
    "PR_EfficientNetV2_M_Model",
]


class EfficientNetV2_M_Model(nn.Module):
    def __init__(self, name="EfficientNetV2_M"):
        super().__init__()

        self.name = name
        self.base = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(1280, 10, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        x = self.sigmoid(x)
        return x


class ResNet50_Model(nn.Module):
    def __init__(self, name="ResNet50"):
        super().__init__()

        self.name = name

        self.base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.base.fc = nn.Linear(2048, 10, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        x = self.sigmoid(x)
        return x


class ConvNeXt_Base_Model(nn.Module):
    def __init__(self, name="ConvNeXT_Base"):
        super().__init__()

        self.name = name

        self.base = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.base.classifier[2] = nn.Linear(1024, 10, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        x = self.sigmoid(x)
        return x


class PR_EfficientNetV2_M_Model(nn.Module):
    def __init__(self, name="PR_EfficientNetV2_M"):
        super().__init__()

        self.name = name

        self.base = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(1280, 10, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raise NotImplementedError()


def init_model(model):
    if isinstance(model, nn.Module):
        return model
    elif isinstance(model, os.PathLike) or (
        type(model) is str and os.path.isfile(model)
    ):
        return torch.load(model)
    elif (type(model) is str) and (str in MODEL_LIST):
        pass
    else:
        raise Exception("Cannot init model.")


def init_optimizer(model: nn.Module):
    raise NotImplementedError()


if __name__ == "__main__":
    m = ConvNeXt_Base_Model("a")
    print(m)
