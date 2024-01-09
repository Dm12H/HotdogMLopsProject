import hydra
import mlflow
from hotdog_mlops.data import HotdogDataset
from torch.utils.data import DataLoader
import os

from hotdog_mlops.model import SqeezeNetClassifier
from hotdog_mlops.trainer import TrainingManager


@hydra.main(
    config_path="config",
    config_name="config.yaml",
    version_base="1.3"
)
def train(cfg):
    mlflow.set_tracking_uri(f"http://{cfg.logging.mlflow_addr}")
    mlflow.set_experiment(cfg.experiment.id)

    base_dir = os.path.dirname(
        hydra.utils.to_absolute_path(__file__)
    )
    cfg.base_dir = base_dir
    experiment_cfg = cfg.experiment
    model = SqeezeNetClassifier(model_params=experiment_cfg.model,
                                transform_params=experiment_cfg.preprocessing)

    train_dset = HotdogDataset(cfg, model.transforms)
    val_dset = HotdogDataset(cfg, model.transforms, dset_type="val")

    train_loader = DataLoader(train_dset,
                              batch_size=experiment_cfg.train.batch,
                              shuffle=experiment_cfg.train.shuffle)

    val_loader = DataLoader(val_dset,
                            batch_size=experiment_cfg.train.batch,
                            shuffle=experiment_cfg.train.shuffle)

    trainer = TrainingManager(experiment_cfg.train)
    trained_model = trainer.train(
        model,
        train_loader,
        val_loader,
        run_name=experiment_cfg.run_name)

    trained_model.save(cfg.model_out_path)


if __name__ == "__main__":
    train()
