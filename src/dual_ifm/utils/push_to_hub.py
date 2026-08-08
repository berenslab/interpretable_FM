"""Push a pretrained SSL backbone checkpoint (SimCLR, t-SimCNE, t-SimCNE-2D) to the
Hugging Face Hub, as a subfolder of a single repo shared by all released variants.

Only for the pretrained backbones in `model_weights/` -- NOT the finetuned downstream
classifiers. Expects a weights-only checkpoint (see `extract_weights.py`), not a full
training checkpoint with optimizer/scheduler state.

Requires `pip install huggingface_hub` and `huggingface-cli login`.

Usage:
    python -m dual_ifm.utils.push_to_hub \
        --checkpoint model_weights/tsimcne2d_bagnet33_all_256_1225.pt \
        --repo-id <your-username>/dual-ifm

This uploads to the `tsimcne2d-bagnet33-256` subfolder of `<your-username>/dual-ifm`.
Load it back with:
    DualIFMBackbone.from_pretrained('<your-username>/dual-ifm', subfolder='tsimcne2d-bagnet33-256')
"""

import argparse
import re
import tempfile
from pathlib import Path

import torch
from huggingface_hub import upload_folder

from dual_ifm.utils.hf_hub import DualIFM

# Checkpoint filename convention: <ssl_method>_<backbone>_<dataset>_<image_size>_<epochs>.pt
NAME_PATTERN = re.compile(
    r'^(?P<ssl_method>simclr|tsimcne2d|tsimcne)_(?P<backbone>[a-z0-9]+)_'
    r'(?P<dataset>[a-z]+)_(?P<image_size>\d+)_(?P<epochs>\d+)$'
)


def parse_checkpoint_name(path):
    stem = Path(path).stem
    match = NAME_PATTERN.match(stem)
    if match is None:
        raise ValueError(
            f'Checkpoint name {stem!r} does not follow the '
            '<ssl_method>_<backbone>_<dataset>_<image_size>_<epochs> convention.'
        )
    return match.groupdict()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='path to a weights-only .pt file (see extract_weights.py)',
    )
    parser.add_argument('--repo-id', required=True, help='e.g. <username>/dual-ifm')
    parser.add_argument(
        '--subfolder',
        default=None,
        help='defaults to <ssl_method>-<backbone>-<image_size>, e.g. tsimcne2d-bagnet33-256',
    )
    parser.add_argument('--private', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    checkpoint_path = Path(args.checkpoint)
    meta = parse_checkpoint_name(checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(state_dict, dict) or 'state_dict' in state_dict:
        raise ValueError(
            'A weights-only checkpoint is expected (see extract_weights.py).'
        )

    model = DualIFM(
        backbone=meta['backbone'],
        ssl_method=meta['ssl_method'],
        img_size=(int(meta['image_size']), int(meta['image_size'])),
    )
    model.load_state_dict(state_dict)

    subfolder = args.subfolder or (
        f'{meta["ssl_method"]}-{meta["backbone"]}-{meta["image_size"]}'
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        upload_folder(
            repo_id=args.repo_id,
            folder_path=tmp_dir,
            path_in_repo=subfolder,
            commit_message=f'Add {subfolder} (pretrained on {meta["dataset"]})',
            repo_type='model',
        )

    print(f'Pushed {checkpoint_path} to {args.repo_id}/{subfolder}')


if __name__ == '__main__':
    main()
