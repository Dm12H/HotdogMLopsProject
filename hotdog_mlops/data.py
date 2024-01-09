from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class HotdogDataset(Dataset):
    def __init__(self, config, transforms, dset_type="train"):
        if dset_type not in {"train", "val"}:
            raise ValueError(f"wrong dset_type '{dset_type}'")
        self.path = Path(
            config.base_dir,
            config.dataset_dir,
            dset_type)
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

    def __getitem__(self, idx):
        img = Image.open(str(self.ims[idx]))
        transformed_img = self.transforms(img)
        label = self.labels[idx]
        return transformed_img, label

    def __len__(self):
        return self._length