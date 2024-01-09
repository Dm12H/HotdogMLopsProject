from torchvision.models import squeezenet1_1,  SqueezeNet1_1_Weights
from torchvision.transforms import v2
from PIL import Image
import torch


class SqeezeNetClassifier:
    def __init__(
            self,
            model_params,
            transform_params=None
    ):
        use_default_weights = model_params["use_default_weights"]
        weights = 'IMAGENET1K_V1' if use_default_weights else None
        self._model = squeezenet1_1(weights)
        self._transforms = self._get_transforms(
            use_default_weights,
            transform_params
        )

    @staticmethod
    def _get_default_transforms():
        return SqueezeNet1_1_Weights.IMAGENET1K_V1.transforms

    @classmethod
    def _get_transforms(cls, use_default_w, params):
        if params is None:
            if use_default_w:
                raise ValueError(
                    "must either use default weights and transforms, "
                    "or provide parameters for your own"
                )
            transforms = cls._get_default_transforms()
        else:
            transforms = cls._build_transformer(params)
        return transforms
    
    def set_transforms(self, transforms):
        self._transforms = transforms
    
    def _inference(self, img, mode):
        if mode == "train":
            assert True
            return self._model(img)

        elif mode == "eval":
            with torch.no_grad():
                transformed_img = self._transforms(img)
                out = self._model(transformed_img)
            return out

        else:
            raise ValueError("supported modes are 'train' and 'eval'")

    def __call__(self, img, mode="eval"):
        return self._inference(img, mode)
