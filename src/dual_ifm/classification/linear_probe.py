import json
import time

import hydra
import torch
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dual_ifm.classification import models
from dual_ifm.utils import datasets, set_seed
from dual_ifm.utils.train import init_eval_experiment


def get_dataloaders(cfg):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name],
    }
    transform = datasets.get_augmentations(
        img_size=cfg.dataset.image_size, normalization=normalization, imagenet=False
    )['test']

    dataset_train, mapping = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=transform,
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=True,
        sample_size=cfg.dataset.sample_size,
        split='train',
        kfold=0,
    )
    dataset_val, mapping_val = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=transform,
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=True,
        sample_size=cfg.dataset.sample_size,
        split='val',
        kfold=0,
    )
    assert mapping_val == mapping, "Val set class mapping differs from train"
    dataset_test, mapping_test = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=transform,
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=True,
        sample_size=cfg.dataset.sample_size,
        split='test',
        kfold=0,
    )
    assert mapping_test == mapping, "Test set class mapping differs from train"
    n_classes = len(mapping)

    # Use all labeled data (train+val) to fit the probe; evaluate on held-out test set
    dataset_train = ConcatDataset((dataset_train, dataset_val))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader_train, dataloader_test, n_classes


def get_model(cfg, n_classes):
    if '.pt' in cfg.model.name:
        model = models.CNNwithProjectorandClassifier(
            img_size=cfg.dataset.image_size,
            backbone=cfg.model.name,
            n_classes=n_classes,
        )

    else:
        model = models.CNNwithClassifier(
            img_size=cfg.dataset.image_size,
            backbone=cfg.model.name,
            weights='DEFAULT',
            n_classes=n_classes,
        )

    return model


@torch.no_grad()
def get_highdim_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()
    X, y = [], []

    print('Computing embeddings...')
    for img, label in tqdm(dataloader, total=len(dataloader)):
        img = img.to(device)

        if isinstance(model, models.CNNwithProjectorandClassifier):
            # Get features before projector
            h_full = model.backbone(img)
            h = model.avgpool(h_full).flatten(start_dim=1)
        else:
            h, _ = model(img)

        X.append(h.cpu())
        y.append(label.cpu())

    X = torch.cat(X).numpy()
    y = torch.cat(y).numpy()

    return X, y


@hydra.main(version_base=None, config_path='../../../configs/classification')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    _, results_file = init_eval_experiment(cfg)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_test, n_classes = get_dataloaders(cfg)
    model = get_model(cfg, n_classes)

    X_train, y_train = get_highdim_embeddings(model, dataloader_train, cfg.device)
    X_test, y_test = get_highdim_embeddings(model, dataloader_test, cfg.device)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        solver=cfg.train.solver,
        penalty=cfg.train.penalty,
        l1_ratio=cfg.train.l1_ratio,
        C=cfg.train.c,
        class_weight=cfg.train.class_weight,
        random_state=cfg.seed,
        max_iter=cfg.train.max_iter,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    preds_test = clf.predict(X_test)
    probs_test = clf.predict_proba(X_test)

    results = {
        'embeddings': X_test.tolist(),
        'targets': y_test.tolist(),
        'preds': preds_test.tolist(),
        'probs': probs_test.tolist(),
    }

    with open(results_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total time: {time.perf_counter() - start:.1f}s')
