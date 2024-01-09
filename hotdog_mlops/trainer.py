import sys
import warnings
from typing import Optional, Union

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .inference import evaluate
from .model import SqeezeNetClassifier
from .utils import current_commit_id

TorchLoss = torch.nn.modules.loss._Loss
TorchScheduler = Union[LRScheduler, ReduceLROnPlateau]


class TrainingManager:
    _losses = {
        "cross_entropy": torch.nn.BCELoss,
        "triplet": torch.nn.TripletMarginLoss,
    }

    _optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    def __init__(self, training_config: DictConfig):
        self.config = training_config
        self.loss_func = self._get_loss_function()
        self.every_step = False

    def _get_loss_function(self) -> TorchLoss:
        loss_config = self.config.loss
        loss_type = self._losses.get(loss_config.name, None)
        if loss_type is None:
            err_msg = f"unsupported or empty loss :'{loss_config.name}'"
            raise TypeError(err_msg, sys.stderr)
        try:
            loss_params = loss_config.get("params", dict())
            loss_func = loss_type(**loss_params)
        except (AttributeError, TypeError) as e:
            loss_name = loss_config.name
            err_msg = f"wrong/missing config values for loss: {loss_name}"
            print(err_msg, sys.stderr)
            raise e

        return loss_func

    def _get_optimizer(self, model) -> torch.optim.Optimizer:
        optimizer_config = self.config.optimizer
        optimizer_name = optimizer_config.name
        opt_type = self._optimizers.get(optimizer_name, None)
        if opt_type is None:
            err_msg = f"unsupported or empty optimizer" f" :'{optimizer_name}'"
            raise TypeError(err_msg, sys.stderr)
        try:
            opt = opt_type(model.parameters(), **optimizer_config.params)
        except (AttributeError, TypeError) as e:
            err_msg = f"wrong/missing params for optimizer: '{optimizer_name}'"
            print(err_msg, sys.stderr)
            raise e
        return opt

    def _get_scheduler(self) -> Optional[TorchScheduler]:
        scheduler_config = self.config.get("scheduler")
        if scheduler_config is None:
            warnings.warn("Scheduler not set, training without one")
            return
        return None

    def train_epoch(
        self,
        model: SqeezeNetClassifier,
        optimizer: torch.optim.Optimizer,
        scheduler: TorchScheduler,
        train_loader,
    ) -> dict[str:float]:
        losses = []
        preds = []
        labels = []

        model.train()
        for data, target in train_loader:
            target = model.encode_labels(target).to(torch.float32)
            batch_size = len(data)
            probs = model.forward(data)

            loss = self.loss_func(probs, target)
            losses.append(loss.item() * batch_size)

            batch_preds = probs.round().tolist()
            batch_labels = target.round().tolist()

            preds += batch_preds
            labels += batch_labels

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        avg_loss = np.mean(losses)
        acc = accuracy_score(y_pred=preds, y_true=labels)
        prec = precision_score(y_pred=preds, y_true=labels)
        recall = recall_score(y_pred=preds, y_true=labels)

        result_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "loss": avg_loss,
        }

        return result_dict

    def train(self, model, train_loader, val_loader, run_name="a run"):
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(dict(self.config))
            mlflow.log_params(dict(model.config))
            mlflow.log_param("commit ID", current_commit_id())

            n_epochs = self.config.n_epochs
            optimizer = self._get_optimizer(model)
            scheduler = self._get_scheduler()
            for epoch in range(n_epochs):
                scheduler_for_epoch = scheduler if self.every_step else None
                train_metrics = self.train_epoch(
                    model, optimizer, scheduler_for_epoch, train_loader
                )
                val_metrics = evaluate(model, val_loader, self.loss_func)

                if scheduler is not None:
                    if not self.every_step:
                        try:
                            scheduler.step(metrics=val_metrics["loss"])
                        except TypeError:
                            scheduler.step()
                    try:
                        last_lr = scheduler.get_last_lr()[0]
                    except AttributeError:
                        last_lr = scheduler._last_lr[0]
                    train_metrics["learning rate"] = last_lr
                else:
                    print(f"Epoch {epoch}")
                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]
                print(f" train loss: {train_loss}")
                print(f" val loss: {val_loss}")

                for metric_name, val in train_metrics.items():
                    mlflow.log_metric(metric_name, val, step=epoch)

                for metric_name, val in val_metrics.items():
                    mlflow.log_metric("val_" + metric_name, val, step=epoch)
            print("training completed")
        return model
