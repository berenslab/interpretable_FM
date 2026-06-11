import json
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from openTSNE import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_ifm.classification import models
from dual_ifm.utils import datasets, plot, set_seed
from dual_ifm.utils.train import check_dead_layers, init_eval_experiment


def get_dataloader(cfg):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name_eval],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name_eval],
    }
    transform = datasets.get_augmentations(
        img_size=cfg.dataset.image_size, normalization=normalization, imagenet=False
    )['test']

    dataset, mapping = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name_eval,
        transform=transform,
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=False,
        sample_size=cfg.dataset.sample_size,
        split='all_holdout',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    n_classes = len(mapping) if mapping.get(np.nan, None) is None else len(mapping) - 1
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
        raise FileNotFoundError(f'No existing checkpoint at {checkpoint_file}')

    return model


@torch.no_grad()
def get_embeddings(model, dataloader, device):
    model.eval()
    X, X_2d, y = [], [], []

    print('Computing embeddings...')
    for _, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = img.to(device)

        if isinstance(model, models.CNNwithProjectorandClassifier):
            # Get features before projector
            h_full = model.backbone(img)
            h = model.avgpool(h_full).flatten(start_dim=1)
            if 'tsimcne' in model.backbone_id:
                z = model.projector(h)
                X_2d.append(z.cpu())
        else:
            h, _ = model(img)

        X.append(h.cpu())
        y.append(label.cpu())

    X = torch.cat(X).numpy()
    X_2d = torch.cat(X_2d).numpy() if X_2d else None
    y = torch.cat(y).numpy()

    return X, X_2d, y


@hydra.main(version_base=None, config_path='../../../configs/classification')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    checkpoint_file, results_file = init_eval_experiment(cfg)
    results_file = results_file.with_name(results_file.stem + '_embeddings.json')
    plot_file = results_file.with_suffix('.svg')

    # Get dataloader and model
    dataloader, n_classes = get_dataloader(cfg)
    model = get_model(cfg, n_classes, checkpoint_file)

    # Check for dead convolutional layers
    check_dead_layers(model)

    # Get embeddings
    X, X_2d, y = get_embeddings(model, dataloader, cfg.device)

    # Encoder embeddings
    if 'tsimcne' not in model.backbone_id:
        tsne = TSNE(
            perplexity=30,
            initialization='pca',
            metric='cosine',
            n_jobs=8,
            random_state=42,
            verbose=False,
        )

        print('Fitting t-SNE...')
        tsne_embeddings = tsne.fit(X)
        X_2d = np.array(tsne_embeddings)

    # Save results
    results = {'x': X.tolist(), 'x_2d': X_2d.tolist(), 'y': y.tolist()}

    with open(results_file, 'w') as f:
        json.dump(results, f)

    # Plot embeddings
    plot.plot_embeddings(
        X=X_2d,
        y=np.expand_dims(y, axis=-1),
        mappings=[{}],
        plot_file=plot_file,
        n_subplots=(1, 1),
        fig_width='full',
        fig_height_ratio=0.6,
        titles=[f'{cfg.dataset.name_eval}: {cfg.dataset.feature}'],
        imbalanced=[False],
        categorical=[False],
        s_marker=30,
    )


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total testing time: {time.perf_counter() - start:.1f} s')
