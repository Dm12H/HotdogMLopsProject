import os
from pathlib import Path

import hydra
import pandas as pd
from torch.utils.data import DataLoader

from hotdog_mlops.data import HotdogDataset
from hotdog_mlops.inference import run_inference
from hotdog_mlops.model import SqeezeNetClassifier


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def infer(cfg):
    base_dir = os.path.dirname(hydra.utils.to_absolute_path(__file__))
    cfg.base_dir = base_dir
    experiment_cfg = cfg.experiment
    model = SqeezeNetClassifier(
        model_params=experiment_cfg.model,
        transform_params=experiment_cfg.preprocessing,
    )
    if os.path.isabs(cfg.model):
        pretrained_model_path = cfg.model
    else:
        pretrained_model_path = Path(base_dir, cfg.model)
    model.load(pretrained_model_path)

    val_dset = HotdogDataset(cfg, model.transforms, dset_type="val", mode="infer")

    val_loader = DataLoader(
        val_dset, batch_size=experiment_cfg.train.batch, shuffle=False
    )

    out_dict = run_inference(model, val_loader)
    df = pd.DataFrame(out_dict)

    if os.path.isabs(cfg.infer_out):
        full_output_path = cfg.infer_out
    else:
        full_output_path = Path(base_dir, cfg.infer_out)

    df.to_csv(full_output_path, sep=";", index=False)
    print(f"Inference finished, results saved to {full_output_path}")


if __name__ == "__main__":
    infer()
