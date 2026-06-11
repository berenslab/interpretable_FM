import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def ddp_setup():
    """DDP setup for multiGPU training scripts."""
    local_rank = int(os.environ['LOCAL_RANK'])  # Provided by torchrun
    torch.cuda.set_device(local_rank)  # rank = unique identifier of each process

    rank = int(os.environ['RANK'])  # global rank
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
    )
    return rank, local_rank, world_size


def init_experiment(cfg, rank=None):
    """Initialize training experiment with hydra and wandb."""
    hydra_cfg = HydraConfig.get()
    experiment_dir = Path(hydra_cfg.runtime.output_dir)

    # Check if it is a sweep
    is_sweep = hydra_cfg.mode.name == 'MULTIRUN'

    # Args to initialize wandb
    if is_sweep:
        run_name = hydra_cfg.job.override_dirname
        group = cfg.sweep_name
    else:
        run_name = cfg.experiment_name
        group = cfg.experiment_name

    wandb_kwargs = dict(
        project='dual_ifm',
        dir=experiment_dir,  # hydra run directory
        name=run_name,
        group=group,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        settings=wandb.Settings(
            console='off',
            save_code=False,  # no code
            disable_git=True,  # no git diffs
            _disable_meta=True,  # no metadata
        ),
    )

    # Initialize checkpoint file separate from logs
    project_dir = Path.cwd()
    checkpoint_file = project_dir.joinpath('checkpoints', group, f'{run_name}.pt')

    if (rank == 0) or (rank is None):
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    return wandb_kwargs, checkpoint_file


def init_eval_experiment(cfg):
    hydra_cfg = HydraConfig.get()

    # Check if it is a sweep
    is_sweep = hydra_cfg.mode.name == 'MULTIRUN'

    if is_sweep:
        run_name = hydra_cfg.job.override_dirname
        group = cfg.sweep_name
    else:
        run_name = cfg.experiment_name
        group = cfg.experiment_name

    # Initialize results file separate from logs
    project_dir = Path.cwd()
    checkpoint_file = project_dir.joinpath('checkpoints', group, f'{run_name}.pt')
    results_file = project_dir.joinpath('checkpoints', group, f'{run_name}.json')
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    return checkpoint_file, results_file


def get_grad_norm(model):
    """Get gradient norm and max from a model."""
    grads = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
    grad_max = max(g.abs().max().item() for g in grads)
    return grad_norm, grad_max


def get_dist_stats(dist):
    """To debug growing distances with tsimcne."""
    q = torch.tensor([0.0, 0.05, 0.5, 0.95, 1], device=dist.device)
    p = torch.quantile(dist, q)

    dist_stats = {
        'p0': p[0].item(),
        'p5': p[1].item(),
        'p50': p[2].item(),
        'p95': p[3].item(),
        'p100': p[4].item(),
    }

    return dist_stats


def reduce(value, local_rank):
    """Reduce a value or tensor across GPUs."""
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=local_rank)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value = value / dist.get_world_size()
    return value


def check_dead_layers(model):
    """Check for suspected dead convolutional layers."""
    dead_layer_count = 0
    for name, parameters in model.named_parameters():
        if 'conv' in name:
            max_weight = parameters.flatten().abs().max()

            if max_weight <= 1e-4:
                dead_layer_count += 1

    print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')
