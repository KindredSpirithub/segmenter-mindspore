import mindspore
import numpy as np
import yaml
from pathlib import Path

IGNORE_LABEL = 255
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}



def dataset_cat_description(path, cmap=None):
    desc = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    colors = {}
    names = []
    for i, cat in enumerate(desc):
        names.append(cat["name"])
        if "color" in cat:
            colors[cat["id"]] = mindspore.Tensor(cat["color"]).float() / 255
        else:
            colors[cat["id"]] = mindspore.Tensor(cmap[cat["id"]]).float()
    colors[IGNORE_LABEL] = mindspore.Tensor([0.0, 0.0, 0.0]).float()
    return names, colors


def rgb_normalize(x, stats):
    """
    x : C x *
    x \in [0, 1]
    """
    return F.normalize(x, stats["mean"], stats["std"])


def rgb_denormalize(x, stats):
    """
    x : N x C x *
    x \in [-1, 1]
    """
    mean = mindspore.Tensor(stats["mean"])
    std = mindspore.Tensor(stats["std"])
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x
