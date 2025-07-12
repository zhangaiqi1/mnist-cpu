"""
存放模型的文件，后续切换模型，只需要修改这里

"""

import torchvision.models as models
import torch.nn as nn


def build_model(name: str, in_ch: int, num_classes: int):
    if name == "resnet18":
        m = models.resnet18(num_classes=num_classes)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    elif name == "mobilenet":
        m = models.mobilenet_v2(num_classes=num_classes)
        m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
    else:
        raise ValueError(name)
    return m
