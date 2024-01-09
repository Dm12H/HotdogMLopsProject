import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .model import SqeezeNetClassifier


def evaluate(model: SqeezeNetClassifier, loader, loss_func):
    losses = []
    preds = []
    labels = []
    model.eval()
    for data, target in loader:
        target = model.encode_labels(target).to(torch.float32)
        batch_size = len(data)
        probs = model.forward(data)

        loss = loss_func(probs, target)
        losses.append(loss.item() * batch_size)

        batch_preds = probs.round().tolist()
        batch_labels = target.tolist()

        preds += batch_preds
        labels += batch_labels

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


def run_inference(model: SqeezeNetClassifier, loader: DataLoader):
    predictions, filenames = [], []
    for data, fnames in loader:
        preds = model.predict(data)

        predictions += list(preds)
        filenames += list(fnames)

    out_dict = {"predicted_labels": predictions, "filenames": filenames}
    return out_dict
