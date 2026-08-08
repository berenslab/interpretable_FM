"""Hugging Face Hub wrapper for the pretrained SSL backbones (SimCLR, t-SimCNE, t-SimCNE-2D).

Covers only the SSL-pretrained backbones released in `model_weights/` -- NOT the finetuned
downstream classifiers (`CNNwithProjectorandClassifier`).

Requires `pip install huggingface_hub`. See `push_to_hub.py` for the upload script.
"""

import json
import os

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_model

from dual_ifm.tsimcne.models import CNNwithProjector

# t-SimCNE and t-SimCNE-2D mutate the projector's last layer down to a 2D in stage 2
EMBEDDING_DIM = {'simclr': 128, 'tsimcne': 2, 'tsimcne2d': 2}


class DualIFM(CNNwithProjector, PyTorchModelHubMixin):
    """Pretrained SSL backbone for retinal fundus images (Dual-IFM).

    Subclasses `CNNwithProjector` so `state_dict()` keys match the original training
    checkpoints exactly (no key remapping needed when loading released weights).

    Args:
        backbone (str): torchvision or BagNet backbone name, e.g. 'bagnet33', 'resnet50'.
        ssl_method (str): one of 'simclr', 'tsimcne', 'tsimcne2d' -- determines embedding_dim.
        img_size (tuple[int, int]): input image size used during pretraining.
    """

    def __init__(
        self, backbone='bagnet33', ssl_method='tsimcne2d', img_size=(256, 256), **kwargs
    ):
        if ssl_method not in EMBEDDING_DIM:
            raise ValueError(
                f'Unknown ssl_method {ssl_method!r}, expected one of {list(EMBEDDING_DIM)}.'
            )
        super().__init__(
            img_size=tuple(img_size),
            backbone=backbone,
            weights=None,
            embedding_dim=EMBEDDING_DIM[ssl_method],
        )
        self.ssl_method = ssl_method

    def forward(self, x):
        """Returns (features, projection) -- see `CNNwithProjector.forward`."""
        return super().forward(x)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        subfolder=None,
        force_download=False,
        token=None,
        cache_dir=None,
        local_files_only=False,
        revision=None,
        **model_kwargs,
    ):
        """Like `PyTorchModelHubMixin.from_pretrained`, but also fetches `config.json`
        from `subfolder` -- the base implementation only checks the repo root, so
        without this override every variant but the one matching the constructor's
        defaults needs `backbone`/`ssl_method` passed in by hand.
        """
        model_id = str(pretrained_model_name_or_path)

        if not os.path.isdir(model_id):
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename='config.json',
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                with open(config_file) as f:
                    for key, value in json.load(f).items():
                        model_kwargs.setdefault(key, value)
            except EntryNotFoundError:
                pass

        return cls._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            subfolder=subfolder,
            **model_kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id,
        revision,
        cache_dir,
        force_download,
        local_files_only,
        token,
        map_location='cpu',
        strict=False,
        subfolder=None,
        **model_kwargs,
    ):
        """Load weights from `subfolder` of a shared repo (one repo, one subfolder per
        variant, uploaded by `push_to_hub.py`). `PyTorchModelHubMixin` has no
        `subfolder` support, so this implements it.
        """
        model = cls(**model_kwargs)

        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, subfolder or '', 'model.safetensors')
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename='model.safetensors',
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )

        load_model(model, model_file, strict=strict, device=map_location)
        model.eval()
        return model
