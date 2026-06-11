import argparse
from pathlib import Path

import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Extract model weights from a full training checkpoint.'
    )

    parser.add_argument(
        '--checkpoint',
        required=True,
        type=str,
        help='checkpoint filename (relative to checkpoints dir) or absolute path',
    )

    parser.add_argument(
        '--output',
        default=None,
        type=str,
        help='output filename for the weights file (default: same name as checkpoint)',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    project_dir = Path.cwd()
    checkpoint_path = project_dir / 'checkpoints' / args.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' not in checkpoint:
        raise KeyError(
            f"No 'state_dict' key found in checkpoint. Keys: {list(checkpoint.keys())}"
        )

    state_dict = checkpoint['state_dict']

    output_name = args.output if args.output else checkpoint_path.name
    weights_dir = project_dir / 'model_weights'
    weights_dir.mkdir(exist_ok=True)
    output_path = weights_dir / output_name

    torch.save(state_dict, output_path)
    print(f'Saved weights to {output_path}')
