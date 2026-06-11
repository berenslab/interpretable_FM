"""Adapted from: https://github.com/berenslab/t-simcne"""

import torch.nn as nn
from torchvision.models import get_model

from dual_ifm.utils import bagnetsv2 as bagnets


class CNNwithProjector(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='resnet18',
        weights=None,
        embedding_dim=128,
        hidden_dim=1024,
    ):
        super().__init__()
        self.backbone_id = backbone
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        if 'bagnet' in backbone:
            self.backbone = bagnets.get_bagnet(backbone, weights=weights)
        else:
            self.backbone = get_model(backbone, weights=weights)

        # Resize first convolutional layer for smaller image sizes and remove first maxpool
        if img_size[0] < 128:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        # Remove last fully connected layer and add projector
        self.n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.init_projector()

    def init_projector(self):
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    def mutate_projector(self, embedding_dim=2):
        self.embedding_dim = embedding_dim
        device = next(self.projector.parameters()).device
        self.projector[-1] = nn.Linear(self.hidden_dim, self.embedding_dim).to(device)

    def freeze(self, stage=0):
        if stage == 1:
            # Everything but the last linear layer is frozen
            self.backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.projector[-1].requires_grad_(True)
        else:
            # Nothing is frozen
            self.backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


if __name__ == '__main__':
    pass
