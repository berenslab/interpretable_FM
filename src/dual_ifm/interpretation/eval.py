import cv2
import matplotlib.patches as patches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dual_ifm.classification import models


def load_model(checkpoint_file, img_size, n_classes, device):
    checkpoint = torch.load(
        checkpoint_file, map_location=torch.device('cpu'), weights_only=False
    )

    backbone = checkpoint['backbone']
    if '.pt' in backbone:
        model = models.CNNwithProjectorandClassifier(
            img_size=img_size, backbone=backbone, n_classes=n_classes
        )
    else:
        model = models.CNNwithClassifier(
            img_size=img_size, backbone=backbone, n_classes=n_classes
        )

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model


def get_patches(heatmap, annotation, img_size=(256, 256), n_patches=16):
    """Extract non overlapping patches from the heatmap and its annotation. Assuming image is square.
    Args:
        heatmap (np.array): class evidence map from the model.
        annotation (np.array): annotation mask (only values between 0 and 1).
        image_size (tuple): image size.
        n_patches (int): number of patches per axis (total number of patches = n_patches**2).
    """

    # Resize heatmap to annotation size
    heatmap = cv2.resize(heatmap, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    annotation = cv2.resize(annotation, dsize=img_size, interpolation=cv2.INTER_NEAREST)

    # Circle mask for the foreground
    circle_mask = cv2.circle(
        np.zeros(img_size),
        (img_size[0] // 2, img_size[1] // 2),
        img_size[0] // 2,
        1,
        -1,
    )
    heatmap = circle_mask * heatmap

    heatmap = torch.from_numpy(heatmap)
    annotation = (annotation > 0).astype(float)
    annotation = torch.from_numpy(annotation)

    # Get activation value for each patch and if it has a lesion
    assert img_size[0] == img_size[1], 'Image must be square.'
    patch_size = img_size[0] // n_patches
    heatmap_patches = (
        heatmap.unfold(0, patch_size, patch_size)
        .unfold(1, patch_size, patch_size)
        .amax(dim=(2, 3))
        .flatten()
    )
    annotation_patches = (
        annotation.unfold(0, patch_size, patch_size)
        .unfold(1, patch_size, patch_size)
        .amax(dim=(2, 3))
        .flatten()
    )

    return heatmap_patches, annotation_patches


def get_precision_score(
    heatmap, annotation, img_size=(256, 256), n_patches=16, threshold=0, topk=12
):
    """Extract non overlapping patches from the heatmap and its annotation. Assuming image is square.
    Args:
        heatmap (torch.tensor): class evidence map from the model.
        annotation (torch.tensor): annotation mask (only values between 0 and 1).
        image_size (tuple): image size.
        n_patches (int): number of patches per axis (total number of patches = n_patches**2).
        threshold (float): threshold to consider a patch active.
        topk (int): k
    """
    heatmap_patches, annotation_patches = get_patches(
        heatmap, annotation, img_size, n_patches
    )

    # Get patches above the threshold
    active_indices = heatmap_patches > threshold
    heatmap_patches = heatmap_patches[active_indices]
    annotation_patches = annotation_patches[active_indices]

    # Sort patches by activation value
    heatmap_patches, sort_indices = torch.sort(heatmap_patches, dim=0, descending=True)
    annotation_patches = annotation_patches[sort_indices]

    scores = []
    for k in range(1, topk + 1):
        annotation_patches_k = annotation_patches[:k]
        scores.append(annotation_patches_k.mean().item())

    return scores


def get_dataset_score(
    model,
    dataset,
    transform,
    img_size=(256, 256),
    n_patches=16,
    topk=10,
    device='cuda:0',
):
    """Get aggregated precision score for an entire dataset.
    Args:
        model: model that returns a heatmap when called with return_heatmap=True.
        dataset (torch.utils.data.Dataset): dataset that returns (image, mask, label).
        transform: image transform applied before passing to the model.
        img_size (tuple): image size used for patch extraction.
        n_patches (int): number of patches per axis.
        topk (int): number of top patches to use for the precision score.
        device (str): device to run the model on.
    """
    # Ensure model is on eval mode and on device
    model.eval()
    model.to(device)

    scores = np.zeros(len(dataset))
    for i, (image, mask, label) in enumerate(dataset):
        with torch.no_grad():
            img = transform(image).unsqueeze(0).to(device)
            heatmap = model(img, return_heatmap=True).squeeze().cpu().numpy()

        mask = np.array(mask.convert('1'), dtype=float)
        score = get_precision_score(
            heatmap[label, :, :],
            mask,
            img_size=img_size,
            n_patches=n_patches,
            topk=topk,
        )
        scores[i] = score[-1]

    return np.nan_to_num(scores, nan=0.0).mean()


def get_bbox_patches(
    heatmap, annotation, ax, img_size=(256, 256), n_patches=16, topk=10
):
    # Get values for each patch
    heatmap_patches, _ = get_patches(heatmap, annotation, img_size, n_patches)
    heatmap_patches, sort_indices = torch.sort(heatmap_patches, dim=0, descending=True)

    # Get coordinates for the patches
    patch_size = img_size[0] // n_patches
    heatmap_coords = [
        (row_idx * patch_size, col_idx * patch_size)
        for row_idx in range(n_patches)
        for col_idx in range(n_patches)
    ]
    heatmap_coords = np.array(heatmap_coords)

    # Get top k coordinates
    sort_indices = sort_indices[:topk]
    heatmap_coords = heatmap_coords[sort_indices]

    for row_idx, col_idx in heatmap_coords:
        rect = patches.Rectangle(
            (col_idx, row_idx),
            patch_size,
            patch_size,
            linewidth=0.3,
            edgecolor='blue',
            facecolor='none',
        )
        ax.add_patch(rect)

    return ax


def drop_nans(X, y):
    if np.isnan(y).any():
        mask = ~np.isnan(y)
        X = X[mask, :]
        y = y[mask]
    return X, y


def get_all_metrics(y_test, y_pred, y_prob):
    acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    if y_prob.shape[1] == 2:
        auroc = roc_auc_score(y_test, y_prob[:, 1])
        auprc = average_precision_score(y_test, y_prob[:, 1])
    else:
        auroc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        auprc = average_precision_score(y_test, y_prob, average='macro')

    return auroc, auprc, acc, kappa


def get_KNN_metrics(X, y, n_neighbors=5, seed=42):
    # Split into train and test
    X, y = drop_nans(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Standarize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    # Get metrics
    auroc, auprc, acc, kappa = get_all_metrics(y_test, y_pred, y_prob)

    return auroc, auprc, acc, kappa


def get_2d_PCA(X):
    pca = PCA(n_components=2)

    # Standarize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Transform
    X_transformed = pca.fit_transform(X_norm)

    return X_transformed
