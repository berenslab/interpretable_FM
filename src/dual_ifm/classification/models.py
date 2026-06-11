"""Adapted from: https://github.com/berenslab/t-simcne"""

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import get_model

from dual_ifm.tsimcne.models import CNNwithProjector
from dual_ifm.utils import bagnetsv2 as bagnets


class CNNwithClassifier(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='resnet18',
        weights=None,
        n_classes=2,
        lambda_penalty=0,
        dropout_rate=0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.backbone_id = backbone
        self.lambda_penalty = lambda_penalty

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

        # Remove last fully connected layer and average pool
        self.n_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.clf = nn.Conv2d(
            self.n_features, self.n_classes, kernel_size=1, stride=1, padding=0
        )

    def freeze_backbone(self, requires_grad=False):
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def freeze_clf(self, requires_grad=False):
        for param in self.clf.parameters():
            param.requires_grad = requires_grad

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x, return_heatmap=False):
        h_full = self.backbone(x)

        # Get embeddings
        h = self.avgpool(h_full).flatten(start_dim=1)

        # Classifier branch
        h_map = self.clf(self.dropout(h_full))
        y = self.avgpool(h_map).flatten(start_dim=1)

        if (self.training) and (self.lambda_penalty > 0):
            l1_penalty = torch.abs(h_map).sum(dim=(1, 2, 3)).mean()
            self.sparsity_penalty = self.lambda_penalty * l1_penalty

        if return_heatmap:
            return h_map
        else:
            return h, y


class CNNwithProjectorandClassifier(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='simclr_resnet50_all_256_1000.pt',
        n_classes=2,
        lambda_penalty=0,
        dropout_rate=0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.backbone_id = backbone
        self.lambda_penalty = lambda_penalty

        # Load the backbone with the projector
        pretrained = CNNwithProjector(
            img_size=img_size,
            backbone=backbone.split(sep='_')[1],
            weights=None,
        )
        if 'tsimcne' in backbone:
            pretrained.mutate_projector()
        checkpoint = torch.load(
            os.path.join('checkpoints', backbone),
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        pretrained.load_state_dict(checkpoint['state_dict'])

        # Extract backbone and projector from pretrained
        self.backbone = pretrained.backbone
        self.projector = pretrained.projector
        self.n_features = pretrained.n_features

        # Remove average pooling with flatten
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # New average pool and classifier (implemented as the sparse bagnet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.clf = nn.Conv2d(
            self.n_features, self.n_classes, kernel_size=1, stride=1, padding=0
        )

    def freeze_backbone(self, requires_grad=False):
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def freeze_projector(self, requires_grad=False):
        for param in self.projector.parameters():
            param.requires_grad = requires_grad

    def freeze_clf(self, requires_grad=False):
        for param in self.clf.parameters():
            param.requires_grad = requires_grad

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x, return_heatmap=False):
        h_full = self.backbone(x)

        # Projector branch
        h = self.avgpool(h_full).flatten(start_dim=1)
        z = self.projector(h)

        # Classifier branch
        h_map = self.clf(self.dropout(h_full))
        y = self.avgpool(h_map).flatten(start_dim=1)

        # Sparsity constraint
        if (self.training) and (self.lambda_penalty > 0):
            l1_penalty = torch.abs(h_map).sum(dim=(1, 2, 3)).mean()
            self.sparsity_penalty = self.lambda_penalty * l1_penalty

        if return_heatmap:
            return h_map
        else:
            return z, y


class EarlyStoppingLoss:
    """
    Args:
        patience (int): number of epochs to wait for improvement.
        min_delta (float): minimum change to qualify as an improvement.
        start_epoch (int): epoch to start the counter.
        model_file (str): filename to save the best model.
        verbose (bool): print a message when a checkpoint is saved and the training is stopped.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.1,
        start_epoch=0,
        model_file='model.pt',
        verbose=True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.model_file = model_file
        self.verbose = verbose

        self.best_metric = np.inf
        self.no_improvement_count = 0
        self.stop_training = False
        self.best_model = None

    def __call__(self, epoch, val_metric, checkpoint):
        if self.check_improvement(val_metric):
            # Modify best model and print stats
            self.best_model = checkpoint['state_dict']
            self.print_stats(val_metric)
            self.save_checkpoint(checkpoint)

            self.no_improvement_count = 0
            self.best_metric = val_metric
        else:
            # Best model is the one previously saved
            checkpoint['state_dict'] = self.best_model
            self.save_checkpoint(checkpoint)

            # Only start counting after start epoch
            if epoch >= self.start_epoch:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(
                        f'Stopping early after {self.patience} epochs with no improvement.'
                    )

    def check_improvement(self, val_metric):
        return val_metric < self.best_metric - self.min_delta

    def print_stats(self, val_metric):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.best_metric:.3f} → {val_metric:.3f}). Saving checkpoint...'
            )

    def save_checkpoint(self, checkpoint):
        """Save checkpoint when validation metric increases."""
        torch.save(checkpoint, self.model_file)


class EarlyStoppingMetric(EarlyStoppingLoss):
    """
    Args:
        patience (int): number of epochs to wait for improvement.
        min_delta (float): minimum change to qualify as an improvement.
        start_epoch (int): epoch to start the counter.
        model_file (str): filename to save the best model.
        verbose (bool): print a message when a checkpoint is saved and the training is stopped.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.1,
        start_epoch=0,
        model_file='model.pt',
        verbose=True,
    ):
        super().__init__(patience, min_delta, start_epoch, model_file, verbose)
        # Initialization of best metric changes
        self.best_metric = -np.inf

    def check_improvement(self, val_metric):
        return val_metric > self.best_metric + self.min_delta

    def print_stats(self, val_metric):
        if self.verbose:
            print(
                f'Validation metric increased ({self.best_metric:.3f} → {val_metric:.3f}). Saving checkpoint...'
            )


def get_loss_weights(y, n_classes):
    counts = torch.bincount(torch.tensor(y).type(torch.int64), minlength=n_classes)
    weights = counts.sum() / (n_classes * counts.float())
    return weights


if __name__ == '__main__':
    pass
