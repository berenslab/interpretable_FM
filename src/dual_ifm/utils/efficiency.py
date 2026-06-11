import argparse
import os
import time
from pathlib import Path

import torch
from tsimcne import models


def get_args():
    parser = argparse.ArgumentParser(description='Self supervised evaluation.')

    parser.add_argument(
        '--checkpoint',
        default='tsimcne2d_bagnet33_all_256_1225.pt',
        type=str,
        help='checkpoint path, relative to checkpoints directory',
    )

    parser.add_argument(
        '--imagesize',
        default=256,
        type=int,
        help='image size, only square images are supported',
    )

    parser.add_argument(
        '--batchsize', default=32, type=int, help='batch size for evaluation'
    )

    parser.add_argument(
        '--mixed',
        default=0,
        type=int,
        choices=[0, 1],
        help='use mixed precision with FP16.',
    )

    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help='device in which evaluation will take place',
    )

    args = parser.parse_args()
    args.imagesize = (args.imagesize, args.imagesize)
    args.mixed = bool(args.mixed)
    return args


def add_experiment_args(args):
    all_args = os.path.basename(args.checkpoint).split(sep='.')[0].split(sep='_')
    args.method = all_args[0]
    args.backbone = all_args[1]
    args.dataset = all_args[2]
    args.imagesize = (int(all_args[3]), int(all_args[3]))
    args.epochs = all_args[4]
    return args


def get_model(args, checkpoint_file):
    model = models.CNNwithProjector(
        img_size=args.imagesize, backbone=args.backbone, weights=None
    )
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device('cpu'), weights_only=False
        )
        model.mutate_projector()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception('No existing checkpoint at ', checkpoint_file)

    return model


if __name__ == '__main__':
    args = get_args()

    # Path to load the model from
    project_dir = Path.cwd()
    checkpoints_dir = project_dir.joinpath('checkpoints')
    checkpoint_file = checkpoints_dir.joinpath(args.checkpoint)

    args = add_experiment_args(args)
    model = get_model(args, checkpoint_file)

    test_input = torch.rand((args.batchsize, 3, args.imagesize[0], args.imagesize[0]))

    model.to(args.device)
    model.eval()
    test_input = test_input.to(args.device)
    n_reps = 100

    dtypes = set(p.dtype for p in model.parameters())
    print(dtypes)

    # Measure latency
    # Warm-up
    for _ in range(20):
        with torch.autocast(
            device_type='cuda', dtype=torch.float16, enabled=args.mixed
        ):
            with torch.no_grad():
                _ = model(test_input)

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_reps):
        with torch.autocast(
            device_type='cuda', dtype=torch.float16, enabled=args.mixed
        ):
            with torch.no_grad():
                _ = model(test_input)

    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / n_reps
    print('Latency (s) = ', latency)

    # Mesure space in memory
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.mixed):
        with torch.no_grad():
            _ = model(test_input)

    peak_memory = torch.cuda.max_memory_allocated()
    peak_memory = peak_memory / 1024**2
    print('Peak memory (MB) = ', peak_memory)
