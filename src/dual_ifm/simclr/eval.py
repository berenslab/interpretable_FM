import json
import os
import time

import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from openTSNE import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_ifm.tsimcne import models
from dual_ifm.utils import datasets, set_seed
from dual_ifm.utils.train import init_eval_experiment


def get_dataloader(cfg):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name_eval],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name_eval],
    }
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(normalization['mean'], normalization['sd']),
        ]
    )

    dataset, _ = datasets.load_dataset_all(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name_eval,
        transform=transform,
        image_size=cfg.dataset.image_size,
        sample_size=cfg.dataset.sample_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    return dataloader


def get_model(cfg, checkpoint_file):
    model = models.CNNwithProjector(
        img_size=cfg.dataset.image_size, backbone=cfg.backbone.name, weights='DEFAULT'
    )
    model.to(cfg.device)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=cfg.device, weights_only=False
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception(f'No existing checkpoint at {checkpoint_file}')

    return model


@torch.no_grad()
def get_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)
    X, y = [], []

    print('Computing embeddings...')
    for _, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # print(f'Batch {b + 1} out of {len(dataloader)}...')
        img = img.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            h, _ = model(img)

        X.append(h.cpu())
        y.append(label.cpu())

    X = torch.cat(X)
    y = torch.cat(y)

    return X, y


def load_eval(checkpoints_dir, experiment_name):
    # Load results
    results_file = checkpoints_dir.joinpath(f'{experiment_name}.json')
    with open(results_file, 'r') as f:
        results = json.load(f)
    X, y = [np.array(results[k]) for k in ['x', 'y']]
    del results

    # Load stats
    checkpoint_file = checkpoints_dir.joinpath(f'{experiment_name}.pt')
    checkpoint = torch.load(
        checkpoint_file, map_location=torch.device('cpu'), weights_only=False
    )
    stats, wandb_id = [checkpoint[k] for k in ['stats', 'wandb_id']]
    del checkpoint

    return X, y, stats, wandb_id


@hydra.main(version_base=None, config_path='../../../configs/pretrain')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    checkpoint_file, results_file = init_eval_experiment(cfg)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader = get_dataloader(cfg)
    model = get_model(cfg, checkpoint_file)

    # Check for dead convolutional layers
    dead_layer_count = 0
    for name, parameters in model.named_parameters():
        if 'conv' in name:
            max_weight = parameters.flatten().abs().max()

            if max_weight <= 1e-4:
                dead_layer_count += 1

    print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

    # Get embeddings
    X, y = get_embeddings(model, dataloader, cfg.device)

    # Compute t-sne embeddings
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
    X = np.array(tsne_embeddings)

    # Save results
    results = {'x': X.tolist(), 'y': y.tolist()}

    with open(results_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total testing time: {time.perf_counter() - start:.1f} s')
