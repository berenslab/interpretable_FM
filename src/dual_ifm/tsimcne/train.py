"""Adapted from: https://github.com/berenslab/t-simcne"""

import os
import time
from collections import defaultdict

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dual_ifm.tsimcne import models
from dual_ifm.tsimcne.loss import InfoNCECauchy
from dual_ifm.utils import datasets, set_seed, train
from dual_ifm.utils import optimizers as optim


def get_dataloader(cfg):
    dataset, _ = datasets.load_dataset_all(
        dataset_dir=cfg.dataset.dir,
        dataset_name=cfg.dataset.name,
        transform=None,
        image_size=cfg.dataset.image_size,
        sample_size=cfg.dataset.sample_size,
    )

    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[cfg.dataset.name],
        'sd': datasets.NORMALIZATION_SD[cfg.dataset.name],
    }
    dataset = datasets.ContrastiveDataset(
        dataset, cfg.dataset.image_size, normalization
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
    return dataloader


def get_optim(cfg, stage, model):
    # Linear scaling for smaller batch sizes
    base_lr = 0.03 * cfg.train.batch_size / 256
    lr = [base_lr, base_lr, base_lr / 1000]
    warmup = [10, 0, 10]

    # For stage 1 the scheduler does nothing
    if stage == 1:
        params = model.projector[-1].parameters()
        optimizer = torch.optim.SGD(
            params, lr=lr[stage], momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1, total_iters=1
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr[stage], momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.CosineAnnealingSchedule(
            optimizer,
            n_epochs=cfg.train.epochs[stage],
            warmup_epochs=warmup[stage],
        )

    return optimizer, scheduler


def get_train_objs(cfg, checkpoint_file, wandb_kwargs):
    # Load model, optimizer, scheduler
    start_epoch, start_stage = 0, 0
    model = models.CNNwithProjector(
        img_size=cfg.dataset.image_size, backbone=cfg.backbone.name, weights='DEFAULT'
    )
    model.to(cfg.device)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=cfg.device, weights_only=False
        )
        start_stage = checkpoint['stage']
        start_epoch = checkpoint['epoch']
        stats = defaultdict(list, checkpoint['stats'])

        if start_stage > 0:
            model.mutate_projector()
        model.load_state_dict(checkpoint['state_dict'])
        model.freeze(start_stage)

        optimizer, scheduler = get_optim(cfg, start_stage, model)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Resume logging to the same id
        wandb_kwargs['id'] = checkpoint['wandb_id']
        wandb_kwargs['resume'] = 'must'
    else:
        optimizer, scheduler = get_optim(cfg, start_stage, model)
        stats = defaultdict(list)

    return (model, optimizer, scheduler, start_stage, start_epoch, stats, wandb_kwargs)


def train_epoch(cfg, model, dataloader, loss_fn, optimizer, run, epoch):
    model.train()
    epoch_loss = 0.0

    start_step = epoch * len(dataloader)
    for b, ((x1, x2), _) in enumerate(dataloader):
        x1, x2 = x1.to(cfg.device), x2.to(cfg.device)
        optimizer.zero_grad()
        logs = {}

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, z = model(torch.cat((x1, x2), dim=0))
            loss, dist_pos, dist_neg = loss_fn(z, cfg.wandb.dist)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Log gradients once per epoch
        if b == (len(dataloader) - 1):
            if cfg.wandb.grad:
                grad_norm, grad_max = train.get_grad_norm(model)
                logs.update(
                    {'grad/norm': grad_norm, 'grad/max': grad_max},
                )

            if cfg.wandb.dist:
                pos_stats = train.get_dist_stats(dist_pos.detach())
                neg_stats = train.get_dist_stats(dist_neg.detach())
                logs.update(
                    {
                        **{f'dist/pos/{k}': v for k, v in pos_stats.items()},
                        **{f'dist/neg/{k}': v for k, v in neg_stats.items()},
                    }
                )

            if logs:
                run.log(logs, step=start_step + b)

    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


@hydra.main(version_base=None, config_path='../../../configs/pretrain')
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Initialize checkpointing and wandb
    wandb_kwargs, checkpoint_file = train.init_experiment(cfg)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader = get_dataloader(cfg)
    model, optimizer, scheduler, start_stage, start_epoch, stats, wandb_kwargs = (
        get_train_objs(cfg, checkpoint_file, wandb_kwargs)
    )
    loss_fn = InfoNCECauchy()

    with wandb.init(**wandb_kwargs) as run:
        # Define metrics: x-axis, other metrics
        run.define_metric('epoch')
        run.define_metric('loss/*', step_metric='epoch')
        run.define_metric('grad/*')
        run.define_metric('dist/*')

        for stage in range(start_stage, len(cfg.train.epochs)):
            # At the start of each stage mutate model and freeze layers
            # if necessary and reset optimizer and scheduler
            if start_epoch == 0:
                if stage == 1:
                    model.mutate_projector()
                model.freeze(stage)  # Freeze the model depending on the stage
                optimizer, scheduler = get_optim(cfg, stage, model)

            for epoch in range(start_epoch, cfg.train.epochs[stage]):
                start_time = time.perf_counter()

                # global epoch across stages for logging purposes
                global_epoch = epoch + sum(cfg.train.epochs[:stage])

                # Train one epoch
                train_loss = train_epoch(
                    cfg,
                    model,
                    dataloader,
                    loss_fn,
                    optimizer,
                    run,
                    global_epoch,
                )
                stats['loss_train'].append(train_loss)
                scheduler.step()

                run.log(
                    {
                        'epoch': global_epoch,
                        'loss/train': train_loss,
                    }
                )

                # Save checkpoint
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'stage': stage,
                    'epoch': epoch + 1,
                    'stats': stats,
                    'wandb_id': run.id,
                }
                torch.save(checkpoint, checkpoint_file)

                print(
                    f'Stage {stage}, epoch {epoch + 1}: '
                    f'train loss {train_loss:.3f}, '
                    f'{time.perf_counter() - start_time:.1f} s',
                    flush=True,
                )

            start_epoch = 0


if __name__ == '__main__':
    start = time.perf_counter()

    # Define resolver
    OmegaConf.register_new_resolver('sum', lambda x: sum(x))

    main()

    print(f'Total training time: {time.perf_counter() - start:.1f} s')
