import os

import torch
import torch.nn as nn
from torchvision.models import (ConvNeXt_Base_Weights, ConvNeXt_Small_Weights,
                                EfficientNet_B0_Weights,
                                EfficientNet_V2_L_Weights,
                                EfficientNet_V2_M_Weights, ResNet50_Weights,
                                convnext_base, convnext_small, efficientnet_b0,
                                efficientnet_v2_l, efficientnet_v2_m, resnet50)

MODEL_LIST = [
    "EfficientNetB0_Model",
    "EfficientNetV2_M_Model",
    "EfficientNetV2_L_Model",
    "ResNet50_Model",
    "ConvNeXt_Base_Model",
    "ConvNeXt_Small_Model",
    "PR_EfficientNetV2_M_Model",
]


class EfficientNetB0_Model(nn.Module):
    def __init__(self, name="EfficientNetB0", use_pretrained=False):
        super().__init__()

        self.name = name
        self.base = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        self.base.classifier[1] = nn.Linear(1280, 10, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x


class EfficientNetV2_M_Model(nn.Module):
    def __init__(self, name="EfficientNetV2_M", use_pretrained=False):
        super().__init__()

        self.name = name
        self.base = efficientnet_v2_m(
            weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        self.base.classifier[1] = nn.Linear(1280, 10, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x


class EfficientNetV2_L_Model(nn.Module):
    def __init__(self, name="EfficientNetV2_L", use_pretrained=False):
        super().__init__()

        self.name = name
        self.base = efficientnet_v2_l(
            weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        self.base.classifier[1] = nn.Linear(1280, 10, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x


class ResNet50_Model(nn.Module):
    def __init__(self, name="ResNet50", use_pretrained=False):
        super().__init__()

        self.name = name

        self.base = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        )
        self.base.fc = nn.Linear(2048, 10, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x


class ConvNeXt_Base_Model(nn.Module):
    def __init__(self, name="ConvNeXT_Base", use_pretrained=False, freeze = False):
        super().__init__()

        self.name = name

        self.base = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        self.tail = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1000, 10, bias=True),
        )

        if freeze :
            for param in self.base.parameters():
                param.requires_grad_(False)

            self.base.features[5].requires_grad_(True)
            self.base.features[6].requires_grad_(True)
            self.base.features[7].requires_grad_(True)
            self.base.avgpool.requires_grad_(True)
            self.base.classifier.requires_grad_(True)

    def forward(self, x):
        x = self.base(x)
        x = self.tail(x)
        return x


class ConvNeXt_Small_Model(nn.Module):
    def __init__(self, name="ConvNeXT_Small", use_pretrained=False):
        super().__init__()

        self.name = name

        self.base = convnext_small(
            weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        self.base.classifier[2] = nn.Linear(768, 10, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x


class PR_EfficientNetV2_M_Model(nn.Module):
    def __init__(self, name="PR_EfficientNetV2_M", use_pretrained=False):
        super().__init__()

        self.name = name

        self.base = efficientnet_v2_m(
            weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        self.base.classifier[1] = nn.Linear(1280, 10, bias=True)

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
