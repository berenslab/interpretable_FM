import os
import time
from collections import defaultdict

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dual_ifm.classification import models
from dual_ifm.utils import datasets, set_seed, train


def get_dataloaders(cfg):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name],
    }
    transform = datasets.get_augmentations(
        img_size=cfg.dataset.image_size, normalization=normalization, imagenet=False
    )
    dataset_train, mapping = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=transform['train'],
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=True,
        sample_size=cfg.dataset.sample_size,
        split='train',
        kfold=cfg.dataset.kfold,
    )
    dataset_val, _ = datasets.load_dataset(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=transform['test'],
        image_size=cfg.dataset.image_size,
        feature_name=cfg.dataset.feature,
        drop_nan=True,
        sample_size=cfg.dataset.sample_size,
        split='val',
        kfold=cfg.dataset.kfold,
    )
    n_classes = len(mapping)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,  # Small dataset
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    return dataloader_train, dataloader_val, n_classes


def get_optim(cfg, model, epoch):
    # Train the first epochs with a frozen backbone
    if epoch < cfg.train.unfreeze_epoch:
        model.freeze_backbone(requires_grad=False)
        model.freeze_clf(requires_grad=True)
        optimizer = torch.optim.AdamW(
            model.clf.parameters(), lr=cfg.optim.lr_head, weight_decay=cfg.optim.wd
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1, total_iters=1
        )
    else:
        model.freeze_backbone(requires_grad=True)
        model.freeze_clf(requires_grad=True)
        params = [
            {
                'params': model.backbone.parameters(),
                'lr': cfg.optim.lr_backbone,
                'weight_decay': cfg.optim.wd,
            },
            {
                'params': model.clf.parameters(),
                'lr': cfg.optim.lr_head,
                'weight_decay': cfg.optim.wd,
            },
        ]

        optimizer = torch.optim.AdamW(params)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1, total_iters=1
        )

    return optimizer, scheduler


def get_train_objs(cfg, checkpoint_file, n_classes, dataloader_train, wandb_kwargs):
    # Load model, optimizer, scheduler and loss function
    start_epoch = 0

    if '.pt' in cfg.model.name:
        model = models.CNNwithProjectorandClassifier(
            img_size=cfg.dataset.image_size,
            backbone=cfg.model.name,
            n_classes=n_classes,
            lambda_penalty=cfg.model.sparsity,
            dropout_rate=cfg.model.dropout,
        )
        model.freeze_projector(requires_grad=False)
    else:
        model = models.CNNwithClassifier(
            img_size=cfg.dataset.image_size,
            backbone=cfg.model.name,
            weights='DEFAULT',
            n_classes=n_classes,
            lambda_penalty=cfg.model.sparsity,
            dropout_rate=cfg.model.dropout,
        )
    model.to(cfg.device)

    # Weighted cross entropy
    loss_weights = (
        get_loss_weights(dataloader_train.dataset.labels, n_classes).to(cfg.device)
        if cfg.train.loss_weighted > 0
        else None
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=loss_weights, label_smoothing=cfg.train.label_smoothing
    )

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=cfg.device, weights_only=False
        )

        start_epoch = checkpoint['epoch']
        stats = defaultdict(list, checkpoint['stats'])
        model.load_state_dict(checkpoint['state_dict'])

        optimizer, scheduler = get_optim(cfg, model, start_epoch)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Resume logging to the same id
        wandb_kwargs['id'] = checkpoint['wandb_id']
        wandb_kwargs['resume'] = 'must'
    else:
        optimizer, scheduler = get_optim(cfg, model, start_epoch)
        stats = defaultdict(list)

    return (model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs)


def top1_acc(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def auroc(logits, y):
    probs = F.softmax(logits.float(), dim=1)

    if probs.shape[1] == 2:
        auc = roc_auc_score(y.numpy(), probs[:, 1].numpy())
    else:
        auc = roc_auc_score(y.numpy(), probs.numpy(), multi_class='ovr')

    return auc


def f1_macro(logits, y):
    preds = torch.argmax(logits, dim=1)
    score = f1_score(y.numpy(), preds.numpy(), average='macro')
    return score


def qwk(logits, y):
    preds = torch.argmax(logits, dim=1)
    score = cohen_kappa_score(y.numpy(), preds.numpy(), weights='quadratic')
    return score


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device):
    model.train()
    epoch_loss = 0.0
    all_logits, all_labels = [], []

    # Freeze batch norm
    model.freeze_bn()

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.type(torch.int64).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, logits = model(x)
            loss = loss_fn(logits, y)

            if model.lambda_penalty > 0:
                loss += model.sparsity_penalty

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        all_logits.append(logits.detach().cpu().float())
        all_labels.append(y.cpu().long())

    epoch_loss = epoch_loss / len(dataloader)
    epoch_metric = auroc(torch.cat(all_logits), torch.cat(all_labels))

    return epoch_loss, epoch_metric


@torch.no_grad()
def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.type(torch.int64).to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, logits = model(x)
            loss = loss_fn(logits, y)

        total_loss += loss.item()
        all_logits.append(logits.cpu().float())
        all_labels.append(y.cpu().long())

    total_loss = total_loss / len(dataloader)
    total_metric = auroc(torch.cat(all_logits), torch.cat(all_labels))

    return total_loss, total_metric


def get_loss_weights(y, n_classes):
    counts = torch.bincount(torch.tensor(y).type(torch.int64), minlength=n_classes)
    weights = counts.sum() / (n_classes * counts.float())
    return weights


@hydra.main(version_base=None, config_path='../../../configs/classification')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Initialize checkpointing and wandb
    wandb_kwargs, checkpoint_file = train.init_experiment(cfg)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_val, n_classes = get_dataloaders(cfg)
    model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs = (
        get_train_objs(cfg, checkpoint_file, n_classes, dataloader_train, wandb_kwargs)
    )

    # Early stopping saves the best model
    stopping_kwargs = dict(
        patience=cfg.stop.patience,
        min_delta=cfg.stop.min_delta,
        start_epoch=cfg.train.unfreeze_epoch,
        model_file=checkpoint_file,
        verbose=True,
    )
    if cfg.stop.monitor == 'auroc':
        early_stopping = models.EarlyStoppingMetric(**stopping_kwargs)
    else:
        early_stopping = models.EarlyStoppingLoss(**stopping_kwargs)

    with wandb.init(**wandb_kwargs) as run:
        # Define metrics: x-axis, other metrics
        run.define_metric('epoch')
        run.define_metric('loss/*', step_metric='epoch', summary='min')
        run.define_metric('auroc/*', step_metric='epoch', summary='max')

        scaler = torch.amp.GradScaler()
        for epoch in range(start_epoch, cfg.train.epochs):
            start_time = time.perf_counter()

            # Unfreeze backbone
            if epoch == cfg.train.unfreeze_epoch:
                print('Unfreezing backbone...')
                optimizer, scheduler = get_optim(cfg, model, epoch)

            # Training
            train_loss, train_metric = train_one_epoch(
                model, dataloader_train, loss_fn, optimizer, scaler, cfg.device
            )
            stats['loss_train'].append(train_loss)
            stats['auroc_train'].append(train_metric)

            # Validation
            val_loss, val_metric = validate_one_epoch(
                model, dataloader_val, loss_fn, cfg.device
            )
            stats['loss_val'].append(val_loss)
            stats['auroc_val'].append(val_metric)
            scheduler.step()

            # Logging
            run.log(
                {
                    'epoch': epoch,
                    'loss/train': train_loss,
                    'loss/val': val_loss,
                    'auroc/train': train_metric,
                    'auroc/val': val_metric,
                }
            )

            # Early stopping
            checkpoint = {
                'state_dict': model.state_dict(),
                'backbone': cfg.model.name,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'stats': stats,
                'wandb_id': run.id,
            }
            stop_value = val_metric if cfg.stop.monitor == 'auroc' else val_loss
            early_stopping(epoch, stop_value, checkpoint)
            if early_stopping.stop_training:
                break

            end_time = time.perf_counter()
            print(
                f'Epoch {epoch + 1}, train loss {train_loss:.3f}, '
                f'val loss {val_loss:.3f}, val auroc {val_metric:.3f}, {end_time - start_time:.1f} s'
            )


if __name__ == '__main__':
    start = time.perf_counter()

    main()

    print(f'Total time: {time.perf_counter() - start:.1f}s')
