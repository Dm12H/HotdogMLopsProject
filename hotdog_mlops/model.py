import warnings
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Iterator

import torch
from dvc.api import DVCFileSystem
from sklearn.preprocessing import LabelEncoder
from torch.nn import Parameter
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from .transform import get_transforms


class SqeezeNetClassifier(torch.nn.Module):
    def __init__(
        self,
        model_params,
        transform_params=None,
    ):
        super().__init__()
        self.config = model_params
        use_default_weights = model_params.use_default_weights
        weights = "IMAGENET1K_V1" if use_default_weights else None
        self._label_encoder = self._get_label_encoder()
        self._model = squeezenet1_1(weights=weights)
        self._add_classifier_head()
        self._fix_gradients()
        self.transforms = get_transforms(transform_params)

    def _get_label_encoder(self) -> LabelEncoder:
        if not isinstance(self.config.classes, Iterable):
            raise TypeError("class list must be iterable")
        encoder = LabelEncoder().fit(self.config.classes)
        return encoder

    def _add_classifier_head(self):
        classifier = torch.nn.Sequential(
            OrderedDict(
                [
                    ("0", torch.nn.Dropout(p=self.config.dropout_p)),
                    ("1", torch.nn.Flatten()),
                    ("2", torch.nn.Linear(86528, 1)),
                    ("3", torch.nn.Sigmoid()),
                ]
            )
        )
        self._model.classifier = classifier

    def _fix_gradients(self):
        last_n = self.config.get("last_n")
        if last_n is None:
            return
        layers = list(self._model.features.named_children())
        for name, child in reversed(layers):
            if last_n > 0:
                last_n -= 1
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self._model.forward(inp).flatten()

    def eval(self):
        self._model.eval()

    def train(self, mode: bool = True):
        self._model.train(mode)

    @staticmethod
    def _get_default_transforms():
        return SqueezeNet1_1_Weights.IMAGENET1K_V1.transforms

    def _get_transforms(self, params):
        if params is None:
            if not self.config.use_default_weights:
                raise ValueError(
                    "must either use default weights and transforms, "
                    "or provide parameters for your own"
                )
            transforms = self._get_default_transforms()
        else:
            transforms = self._build_transformer(params)
        return transforms

    def set_transforms(self, transforms):
        self.transforms = transforms

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self._model.parameters(recurse)

    def predict(self, img):
        transformed_img = self.transforms(img)
        out = self._model(transformed_img)
        class_ids = out.flatten().round().to(torch.int32).tolist()
        labels = self._label_encoder.inverse_transform(class_ids)
        return labels

    def __call__(self, img, mode="eval"):
        return self._inference(img, mode)

    def encode_labels(self, labels: Iterable[str]) -> torch.Tensor:
        label_list = self._label_encoder.transform(labels)
        tensor = torch.IntTensor(label_list)
        return tensor

    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load(self, path: Path):
        if not path.exists():
            warnings.warn("model weights are missing, downloading from dvc")
            fs = DVCFileSystem(rev="main")
            fs.get_file(path.name, path)
        self._model.load_state_dict(torch.load(path))
