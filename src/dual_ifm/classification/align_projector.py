import os
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dual_ifm.classification import models
from dual_ifm.tsimcne.loss import InfoNCECauchy
from dual_ifm.utils import datasets, set_seed, train
from dual_ifm.utils import optimizers as optim


def get_dataloader(cfg):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name],
    }

    dataset, mapping = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=None,
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=False,
        sample_size=cfg.dataset.sample_size,
        split='all_holdout',
        kfold=0,
    )

    dataset = datasets.ContrastiveDataset(
        dataset, cfg.dataset.image_size, normalization, 'fundus'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    n_classes = len(mapping) if mapping.get(np.nan, None) is None else len(mapping) - 1
    return dataloader, n_classes


def load_model(cfg, n_classes, checkpoint_file):
    checkpoint = torch.load(
        checkpoint_file, map_location=cfg.device, weights_only=False
    )
    backbone = checkpoint['backbone']
    model = models.CNNwithProjectorandClassifier(
        img_size=cfg.dataset.image_size, backbone=backbone, n_classes=n_classes
    )
    model.to(cfg.device)
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint


def get_train_objs(cfg, n_classes, checkpoint_file, wandb_kwargs):
    checkpoint_file_original = Path.cwd().joinpath('checkpoints', cfg.model.name)
    # Load finetuned weights or resume training
    if os.path.exists(checkpoint_file):
        # Resume alignment
        model, checkpoint = load_model(cfg, n_classes, checkpoint_file)

        # Freeze everything but projector
        model.freeze_backbone(requires_grad=False)
        model.freeze_projector(requires_grad=True)
        model.freeze_clf(requires_grad=False)

        # Load stats, optimizer and scheduler
        start_epoch = checkpoint['epoch']
        stats = defaultdict(list, checkpoint['stats'])

        optimizer, scheduler = get_optim(cfg, model)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Resume logging to the same id
        wandb_kwargs['id'] = checkpoint['wandb_id']
        wandb_kwargs['resume'] = 'must'

    elif os.path.exists(checkpoint_file_original):
        # Start from finetuned weights
        model, _ = load_model(cfg, n_classes, checkpoint_file_original)

        # Freeze everything but projector
        model.freeze_backbone(requires_grad=False)
        model.freeze_projector(requires_grad=True)
        model.freeze_clf(requires_grad=False)

        start_epoch = 0
        optimizer, scheduler = get_optim(cfg, model)
        stats = defaultdict(list)

    else:
        raise Exception(f'No existing checkpoint at {checkpoint_file_original}.')

    return model, optimizer, scheduler, start_epoch, stats, wandb_kwargs


def get_optim(cfg, model):
    # Only the projector is trained
    params = model.projector.parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.CosineAnnealingSchedule(
        optimizer, final_lr=1e-5, n_epochs=cfg.train.epochs, warmup_epochs=5
    )
    return optimizer, scheduler


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    # Only projector in training mode
    model.projector.train()
    model.backbone.eval()
    model.clf.eval()

    epoch_loss = 0.0
    for _, ((x1, x2), _) in enumerate(dataloader):
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Only compute projector branch
            with torch.no_grad():
                h_full = model.backbone(torch.vstack((x1, x2)))
                h = model.avgpool(h_full).flatten(start_dim=1)

            z = model.projector(h)
            loss, _, _ = loss_fn(z)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


@hydra.main(version_base=None, config_path='../../../configs/classification')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Initialize checkpointing and wandb
    wandb_kwargs, checkpoint_file = train.init_experiment(cfg)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader, n_classes = get_dataloader(cfg)
    model, optimizer, scheduler, start_epoch, stats, wandb_kwargs = get_train_objs(
        cfg, n_classes, checkpoint_file, wandb_kwargs
    )
    loss_fn = InfoNCECauchy()

    ###################### TRAINING LOOP #########################
    with wandb.init(**wandb_kwargs) as run:
        # Define metrics: x-axis, other metrics
        run.define_metric('epoch')
        run.define_metric('loss/*', step_metric='epoch')

        for epoch in range(start_epoch, cfg.train.epochs):
            start_time = time.perf_counter()

            # Train one epoch
            train_loss = train_epoch(model, dataloader, loss_fn, optimizer, cfg.device)
            stats['loss_train'].append(train_loss)
            scheduler.step()

            # Logging
            run.log(
                {
                    'epoch': epoch,
                    'loss/train': train_loss,
                }
            )

            # Checkpointing
            checkpoint = {
                'state_dict': model.state_dict(),
                'backbone': model.backbone_id,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'stats': stats,
                'wandb_id': run.id,
            }
            torch.save(checkpoint, checkpoint_file)

            print(
                f'Epoch {epoch + 1}: train loss {train_loss:.3f}, '
                f'{time.perf_counter() - start_time:.1f} s'
            )


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total training time: {time.perf_counter() - start:.1f} s')
