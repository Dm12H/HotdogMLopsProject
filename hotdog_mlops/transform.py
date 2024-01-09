from torchvision.transforms import v2
from omegaconf import DictConfig
import torch


class TransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self):
        return self.transform


class TransformerBuilder:
    _class_map = {
        "resize": v2.Resize,
        "centercrop": v2.CenterCrop,
        "rescale": TransformWrapper(
            v2.Compose(
                [v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True)]
            )
        ),
        "normalize": v2.Normalize
    }

    def __init__(self, config):
        self.config = config

    def build_transforms(self):
        transforms = []
        for tr in self.config.steps:
            tr_type = self._class_map.get(tr)
            if tr_type is None:
                raise ValueError(f"transform '{tr}' not supported")
            attrs = self.config.get(tr, dict())
            transforms.append(tr_type(**attrs))
        composition = v2.Compose(transforms)
        return composition


def get_transforms(config: DictConfig) -> v2.Transform:
    builder = TransformerBuilder(config)
    transforms = builder.build_transforms()
    return transforms
