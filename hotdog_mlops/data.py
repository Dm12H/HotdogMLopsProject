import warnings
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from dvc.api import DVCFileSystem


class HotdogDataset(Dataset):
    def __init__(self, config, transforms, dset_type="train", mode="train"):
        if dset_type not in {"train", "val"}:
            raise ValueError(f"wrong dset_type '{dset_type}'")
        self.path = Path(
            config.base_dir,
            config.dataset_dir,
            dset_type)

        if not self.path.exists():
            fs = DVCFileSystem(rev="main")
            artifact_name = self.path.parent.stem
            warnings.warn(
                "data is missing, downloading from dvc"
            )
            fs.get(
                f"{artifact_name}/{dset_type}",
                str(self.path),
                recursive=True
            )

        hotdog_path = self.path / "hot_dog"
        not_hotdog_path = self.path / "not_hot_dog"

        hotdog_ims = list(hotdog_path.iterdir())
        not_hotdog_ims = list(not_hotdog_path.iterdir())
        self.ims = hotdog_ims + not_hotdog_ims

        hotdog_labels = ["hot_dog"] * len(hotdog_ims)
        not_hotdog_labels = ["not_hot_dog"] * len(not_hotdog_ims)
        self.labels = hotdog_labels + not_hotdog_labels

        self._length = len(self.ims)
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx):
        img = Image.open(str(self.ims[idx]))
        transformed_img = self.transforms(img)
        label = self.labels[idx]
        if self.mode == "train":
            return transformed_img, label
        else:
            name = self.ims[idx].name
            return transformed_img, name

    def __len__(self):
        return self._length