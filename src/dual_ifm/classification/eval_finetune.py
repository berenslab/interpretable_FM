import json
import os
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_ifm.classification import models
from dual_ifm.utils import datasets, set_seed
from dual_ifm.utils.train import check_dead_layers, init_eval_experiment


def get_dataloader(cfg):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name_eval],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name_eval],
    }
    transform = transform = datasets.get_augmentations(
        img_size=cfg.dataset.image_size, normalization=normalization, imagenet=False
    )['test']

    dataset_test, mapping = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name_eval,
        transform=transform,
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=True,
        sample_size=cfg.dataset.sample_size,
        split='test',
    )

    dataloader = DataLoader(
        dataset_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    n_classes = len(mapping)
    return dataloader, n_classes


def get_model(cfg, n_classes, checkpoint_file):
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=cfg.device, weights_only=False
        )

        backbone = checkpoint['backbone']
        if '.pt' in backbone:
            model = models.CNNwithProjectorandClassifier(
                img_size=cfg.dataset.image_size, backbone=backbone, n_classes=n_classes
            )
        else:
            model = models.CNNwithClassifier(
                img_size=cfg.dataset.image_size,
                backbone=backbone,
                n_classes=n_classes,
            )

        model.load_state_dict(checkpoint['state_dict'])
        model.to(cfg.device)
    else:
        raise Exception(f'No existing checkpoint at {checkpoint_file}')

    return model


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()

    preds, probs, targets, embeddings = [], [], [], []
    for b, (img, labels) in enumerate(tqdm(dataloader)):
        img, labels = img.to(device), labels.to(device)

        z, y = model(img)
        pred = torch.argmax(y, dim=1)
        p = F.softmax(y, dim=1)

        preds.append(pred.cpu())
        probs.append(p.cpu())
        targets.append(labels.cpu())
        embeddings.append(z.cpu())

    preds = torch.cat(preds).numpy()
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    embeddings = torch.cat(embeddings).numpy()

    return preds, probs, targets, embeddings


def load_eval(checkpoints_dir, experiment_name, load_stats=True):
    # Load results
    results_file = checkpoints_dir.joinpath(f'{experiment_name}.json')
    with open(results_file, 'r') as f:
        results = json.load(f)

    preds, probs, targets = [
        np.array(results[k]) for k in ['preds', 'probs', 'targets']
    ]
    del results

    # Load stats
    stats = {}
    wandb_id = None
    if load_stats:
        checkpoint_file = checkpoints_dir.joinpath(f'{experiment_name}.pt')
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device('cpu'), weights_only=False
        )
        stats, wandb_id = [checkpoint[k] for k in ['stats', 'wandb_id']]
        del checkpoint

    return preds, probs, targets, stats, wandb_id


def load_embeddings(checkpoints_dir, experiment_name):
    # Load results
    results_file = checkpoints_dir.joinpath(f'{experiment_name}_embeddings.json')
    with open(results_file, 'r') as f:
        results = json.load(f)

    X, X_2d, y = [np.array(results[k]) for k in ['x', 'x_2d', 'y']]
    del results

    return X, X_2d, y


def get_all_metrics(preds, probs, targets):
    # Get metrics
    acc = balanced_accuracy_score(targets, preds)
    kappa = cohen_kappa_score(targets, preds, weights='quadratic')

    if probs.ndim == 1:
        auroc = roc_auc_score(targets, probs)
        auprc = average_precision_score(targets, probs)
    elif probs.shape[1] == 2:
        auroc = roc_auc_score(targets, probs[:, 1])
        auprc = average_precision_score(targets, probs[:, 1])
    else:
        auroc = roc_auc_score(targets, probs, multi_class='ovr')
        auprc = average_precision_score(targets, probs, average='macro')

    return auroc, auprc, acc, kappa


@hydra.main(version_base=None, config_path='../../../configs/classification')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    checkpoint_file, results_file = init_eval_experiment(cfg)

    # Get dataloader and model
    dataloader, n_classes = get_dataloader(cfg)
    model = get_model(cfg, n_classes, checkpoint_file)

    # Check for dead convolutional layers
    check_dead_layers(model)

    # Get predictions (embeddings from the projector)
    preds, probs, targets, embeddings = predict(model, dataloader, cfg.device)

    # Save results
    results = {
        # 'embeddings': embeddings.tolist(),
        'preds': preds.tolist(),
        'probs': probs.tolist(),
        'targets': targets.tolist(),
    }

    with open(results_file, 'w') as f:
        json.dump(results, f)

    # Metrics
    auroc, auprc, acc, kappa = get_all_metrics(preds, probs, targets)
    print(f'AUROC (test) = {auroc:.3f}')
    print(f'AUPRC (test) = {auprc:.3f}')
    print(f'Accuracy (test) = {acc:.3f}')
    print(f'KAPPA (test) = {kappa:.3f}')


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total testing time: {time.perf_counter() - start:.1f} s')
