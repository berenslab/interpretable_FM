import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import CIFAR10, Imagenette
from tqdm import tqdm

# Normalization per dataset
NORMALIZATION_MEAN = {
    'imagenet': [0.485, 0.456, 0.406],
    'imagenette': [0.485, 0.456, 0.406],
    'cifar10': [0.485, 0.456, 0.406],
    'eyepacs': [0.340, 0.215, 0.139],
    'areds': [0.497, 0.279, 0.136],
    'ukb': [0.562, 0.269, 0.096],
    'all': [0.397, 0.233, 0.131],
    'idrid': [0.447, 0.216, 0.069],
    'aptos': [0.376, 0.201, 0.063],
    'messidor': [0.476, 0.221, 0.076],
    'deepdrid': [0.429, 0.263, 0.158],
    'glaucoma': [0.695, 0.418, 0.192],
    'papila': [0.324, 0.111, 0.056],
    'fives': [0.34, 0.157, 0.065],
    'idridncc': [0.478, 0.231, 0.065],
}
NORMALIZATION_SD = {
    'imagenet': [0.229, 0.224, 0.225],
    'imagenette': [0.229, 0.224, 0.225],
    'cifar10': [0.229, 0.224, 0.225],
    'eyepacs': [0.268, 0.189, 0.153],
    'areds': [0.346, 0.215, 0.142],
    'ukb': [0.356, 0.184, 0.087],
    'all': [0.309, 0.194, 0.143],
    'idrid': [0.306, 0.163, 0.083],
    'aptos': [0.288, 0.158, 0.078],
    'messidor': [0.297, 0.149, 0.065],
    'deepdrid': [0.309, 0.202, 0.147],
    'glaucoma': [0.17, 0.158, 0.145],
    'papila': [0.247, 0.092, 0.046],
    'fives': [0.243, 0.125, 0.06],
    'idridncc': [0.315, 0.166, 0.087],
}


def load_image(filename, return_np=True):
    """Load image file as a PIL image or a numpy array."""
    image = Image.open(filename)
    image.load()
    data = np.asarray(image, dtype='int32')
    return data if return_np else image


def get_matching_index(arr_ref, arr_match):
    """Get indices for elements in arr_ref that match elements in arr_match."""
    df = pd.DataFrame({'Value': np.arange(arr_ref.size)}, index=arr_ref)
    arr_match = arr_match[np.isin(arr_match, arr_ref)]
    df = df.loc[arr_match]
    return df['Value'].to_numpy()


class ContrastiveDataset(Dataset):
    def __init__(
        self, dataset: Dataset, img_size: tuple, normalization=None, kind='fundus'
    ):
        self.dataset = dataset
        self.transform = self.get_augmentations(img_size, normalization, kind)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img1 = self.transform(img)
        img2 = self.transform(img)

        return (img1, img2), label

    def get_augmentations(self, img_size, normalization, kind=False):
        if kind == 'imagenet':
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                    transforms.ToTensor(),
                ]
            )
        elif kind == 'fundus':
            # Augmentations for fundus images
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1.0)),
                    transforms.RandomRotation(
                        degrees=(-15, 15),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.05)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                ]
            )
        elif kind == 'fundus-mild':
            # Augmentations for fundus images
            transform = transforms.Compose(
                [
                    transforms.RandomRotation(
                        degrees=(-15, 15),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.2, 0.2, 0.05, 0)], p=0.5
                    ),
                    transforms.ToTensor(),
                ]
            )

        if normalization is not None:
            normalization = transforms.Normalize(
                normalization['mean'], normalization['sd']
            )
            transform.transforms.append(normalization)

        return transform


def get_augmentations(img_size, normalization=None, imagenet=False):
    """Get augmentations for supervised training."""
    if imagenet:
        transform = {
            # Regular augmentations for imagenet
            'train': transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            'test': transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                ]
            ),
        }
    else:
        # Augmentations for fundus images
        transform = {
            'train': transforms.Compose(
                [
                    transforms.RandomRotation(
                        degrees=(-15, 15),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.2, 0.2, 0.05, 0)], p=0.5
                    ),
                    transforms.ToTensor(),
                ]
            ),
            'test': transforms.Compose([transforms.ToTensor()]),
        }

    if normalization:
        normalize = transforms.Normalize(normalization['mean'], normalization['sd'])
        _ = [transform[k].transforms.append(normalize) for k in transform.keys()]

    return transform


def load_cifar10(transform=None, dataset_dir='./datasets'):
    """Load cifar10 dataset, to run tests."""
    dataset_train = CIFAR10(dataset_dir, train=True, download=True, transform=transform)
    dataset_test = CIFAR10(dataset_dir, train=False, transform=transform)
    mapping = {
        0: 'plane',
        1: 'car',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }
    return dataset_train, dataset_test, mapping


def load_imagenette(transform=None, image_size=(224, 224)):
    resize_transform = transforms.Resize(image_size)
    if transform is None:
        transform = resize_transform
    else:
        transform.transforms.insert(0, resize_transform)

    dataset_train = Imagenette(
        'datasets', split='train', transform=transform, size='320px', download=True
    )
    dataset_val = Imagenette(
        'datasets', split='val', transform=transform, size='320px', download=True
    )
    mapping = {i: cl[0] for i, cl in enumerate(dataset_train.classes)}
    return dataset_train, dataset_val, mapping


class EyePACS(Dataset):
    """
    Loads images and labels from EyePACS pre-processed dataset.
    """

    FEATURE_MAP = {
        'patient_gender': 'gender',
        'patient_age': 'age',
        'patient_ethnicity': 'ethnicity',
        # 'diagnosis_image_dr_level': 'dr',
        'dr_bin': 'dr',
        'diagnosis_dme': 'dme',
        'image_field': 'field',
        'clinical_siteIdentifier': 'camera',
        'session_image_quality': 'quality',
        'patient_id': 'patient_id',
    }

    CAT_FEAT = ['gender', 'ethnicity', 'dr', 'dme', 'field', 'camera', 'quality']

    MISSING_TOKENS = ['missing', 'not specified', 'decline', 'unknown', 'other', 'NaN']

    def __init__(
        self,
        image_dir,
        labels_file,
        camera_file,
        transform=None,
        feature_name='age',
        drop_nan=False,
        sample_size=20000,
        split='all',
        kfold=0,
    ):
        """
        Args:
            image_dir (str, Path): path to the image directory.
            labels_file (string, Path): path to the csv file with labels.
            transform (object): image transform.
            feature_name (str ir list[str]): names of features to retrieve from the labels file.
            drop_nan (bool): whether to drop nan values when retrieving a feature.
            split (str): "train", "val", "test" or "all" for kfold=0, otherwise use "train" or "test" only.
            kfold (int): integer from 0 to 5. Use 1 to 5 to get the 5-fold splits and 0 to get train-val-test split.
        """
        self.transform = transform
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.camera_file = camera_file
        self.sample_size = sample_size

        #  Get image paths with specific features
        self.feature_name = feature_name

        self.image_paths, self.labels, self.mappings = self.get_dataset_objects(
            feature_name, split, kfold, drop_nan
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def preprocess_labels(self, df):
        df = self.get_camera_info(df)
        df = df[df['field'].isin(['field 1', 'field 2', 'field 3'])]
        df = df[df['quality'].isin(['Good', 'Excellent', 'Adequate'])]

        # Temporary, image paths are jpeg
        # df['image_path'] = df['image_path'].str.replace(
        #     r'\.(png|jpg|jpeg|PNG|JPG)$', '.jpg', regex=True
        # )
        return df

    def get_dataset_objects(self, feature_names, split, kfold, drop_nan=False):
        """Get image paths and feature values from the labels file."""
        feature_map = self.FEATURE_MAP
        feature_names = (
            feature_names if isinstance(feature_names, list) else [feature_names]
        )

        # Get labels
        df = pd.read_csv(
            self.labels_file,
            usecols=['image_path', 'patient_id'] + list(feature_map.keys()),
        )
        df = df.reset_index(drop=True)
        df = df.rename(columns=feature_map)

        df = self.preprocess_labels(df)

        # Split dataset
        df['patient_id'] = df['patient_id'].astype('str')
        df = df.set_index('patient_id')
        ids = self.get_ids(split, kfold)
        df = df.loc[ids]

        df, mappings = self.get_metadata(feature_names, df, drop_nan)
        df = df.dropna(subset=feature_names) if drop_nan else df

        if self.sample_size is not None:
            df = df.sample(n=self.sample_size, random_state=42)

        features = df[feature_names].to_numpy(dtype=np.float32)
        features = features.squeeze() if len(feature_names) == 1 else features
        image_paths = df['image_path'].to_numpy()
        return image_paths, features, mappings

    def get_ids(self, split, kfold):
        """Get participant ids for the chosen split."""
        assert (
            (kfold == 0) and (split in ['train', 'val', 'test', 'all', 'all_holdout'])
        ) or ((kfold in list(range(1, 6))) and (split in ['train', 'val', 'test'])), (
            'Invalid combination of split and kfold.'
        )

        splits_file = os.path.join(os.path.dirname(self.labels_file), 'splits.json')
        with open(splits_file, 'r') as f:
            splits = json.load(f)

        if (split == 'all') or (split == 'all_holdout'):
            holdout_file = os.path.join(
                os.path.dirname(self.labels_file), 'holdout.json'
            )
            with open(holdout_file, 'r') as f:
                holdout = json.load(f)
            if split == 'all_holdout':
                ids = np.array(holdout['holdout'])
            else:
                ids = np.array(holdout['pretrain'])
        elif split == 'test':
            ids = np.array(splits['0']['test'])
        else:
            ids = np.array(splits[str(kfold)][split])

        ids = ids.astype(str)
        ids.sort()
        return ids

    def get_metadata(self, feature_names, df, drop_nan):
        mappings = []
        for feature_name in feature_names:
            # Encode categorical features
            if feature_name in self.CAT_FEAT:
                feature = pd.Categorical(df[feature_name], ordered=True)
                df[feature_name] = feature.codes
                # Replace -1 with nans for missing values
                df[feature_name] = df[feature_name].replace(-1, np.nan)
                categories = feature.categories.tolist()
                mapping = dict(zip(range(len(categories)), categories))
                mapping = (
                    mapping | {np.nan: 'nocat'}
                    if df[feature_name].isna().any()
                    else mapping
                )
                df, mapping = self.group_nans(feature_name, mapping, df)
            else:
                mapping = {}

            if drop_nan:
                mapping.pop(np.nan, None)

            mappings.append(mapping)

        mappings = mappings[0] if len(feature_names) == 1 else mappings
        return df, mappings

    def group_nans(self, feature_name, mapping, df=None):
        """
        Group missing values under a single nan value
        if they exist for a feature.
        """
        missing = []
        for k, v in mapping.items():
            if isinstance(v, str):
                v = v.lower()
                if any(x in v for x in self.MISSING_TOKENS):
                    if df is not None:
                        df[feature_name] = df[feature_name].replace(k, np.nan)
                    missing.append(k)
                elif 'nocat' in v:
                    mapping[np.nan] = 'Missing'
        if missing:
            mapping[np.nan] = 'Missing'
            _ = [mapping.pop(k) for k in missing]
        return df, mapping

    def get_camera_info(self, df):
        df_camera = pd.read_csv(self.camera_file)
        camera_mapping = dict(
            zip(df_camera['site_id'].to_list(), df_camera['camera'].to_list())
        )
        df['camera'] = df['camera'].map(camera_mapping)
        return df

    def get_feature(self, image_paths, feature_names, drop_nan=False):
        """Load a feature after the dataset has been instantiated."""
        feature_map = self.FEATURE_MAP
        feature_names = (
            feature_names if isinstance(feature_names, list) else [feature_names]
        )

        # Get labels
        df = pd.read_csv(
            self.labels_file, usecols=['image_path'] + list(feature_map.keys())
        )
        df = df.reset_index(drop=True)
        df = df.rename(columns=feature_map)

        df = self.preprocess_labels(df)

        # Get features for the image paths provided
        df = df.set_index('image_path')
        df = df.loc[image_paths]

        df, mappings = self.get_metadata(feature_names, df, drop_nan)
        df = df.dropna(subset=feature_names) if drop_nan else df

        features = df[feature_names].to_numpy(dtype=np.float32)
        features = features.squeeze() if len(feature_names) == 1 else features
        return features, mappings


class AREDS(EyePACS):
    """
    Loads images from AREDS pre-processed dataset.
    """

    FEATURE_MAP = {
        'image_side': 'side',
        'visit_number': 'visit',
        'image_field': 'field',
        'patient_sex': 'gender',
        'patient_age': 'age',
        'clinical_high_blood_pressure_baseline': 'hbp',
        # 'diagnosis_amd_grade': 'amd',
        'clinical_smoking_status': 'smoking',
        'clinical_bmi_status': 'bmi',
        'clinical_diabetes_status': 'diabetes',
        'quality_label': 'quality',
        'amd_grouped': 'amd',
    }

    CAT_FEAT = [
        'side',
        'visit',
        'diabetes',
        'amd',
        'field',
        'gender',
        'smoking',
        'hbp',
        'amdgrouped',
    ]

    def __init__(
        self,
        image_dir,
        labels_file,
        transform=None,
        feature_name='age',
        drop_nan=False,
        sample_size=20000,
        split='all',
        kfold=0,
    ):
        """
        Args:
            image_dir (str, Path): path to the image directory.
            labels_file (string, Path): path to the csv file with labels.
            transform (object): image transform.
            feature_name (str ir list[str]): names of features to retrieve
                                            from the labels file.
            drop_nan (bool): whether to drop nan values when retrieving a feature.
        """
        super().__init__(
            image_dir,
            labels_file,
            camera_file=None,
            transform=transform,
            feature_name=feature_name,
            drop_nan=drop_nan,
            sample_size=sample_size,
            split=split,
            kfold=kfold,
        )

    def preprocess_labels(self, df):
        df = df[df['quality'] == 1]
        return df


class UKB(EyePACS):
    """
    Loads images from AREDS pre-processed dataset.
    """

    FEATURE_MAP = {
        'quality_label': 'quality',
        'image_eye': 'side',
        'quality_score': 'age',  # Temporary
    }

    CAT_FEAT = ['quality', 'side', 'visit']

    def __init__(
        self,
        image_dir,
        labels_file,
        transform=None,
        feature_name=None,
        drop_nan=False,
        sample_size=20000,
        split='all',
        kfold=0,
    ):
        """
        Args:
            image_dir (str, Path): path to the image directory.
            labels_file (string, Path): path to the csv file with labels.
            transform (object): image transform.
            feature_name (str ir list[str]): names of features to retrieve from the labels file.
            drop_nan (bool): whether to drop nan values when retrieving a feature.
        """
        super().__init__(
            image_dir,
            labels_file,
            camera_file=None,
            transform=transform,
            feature_name=feature_name,
            drop_nan=drop_nan,
            sample_size=sample_size,
            split=split,
            kfold=kfold,
        )

    def preprocess_labels(self, df):
        df = df[df['quality'] == 1]
        return df


class IDRiD(EyePACS):
    """
    Loads images from IDRiD pre-processed dataset.
    """

    FEATURE_MAP = {
        'age': 'age',  # temporary
        'Retinopathy grade': 'dr',
        'Risk of macular edema': 'macular_edema',
        'drbin': 'drbin',
    }

    CAT_FEAT = ['dr', 'drbin', 'macular_edema']

    def __init__(
        self,
        image_dir,
        labels_file,
        transform=None,
        feature_name='age',
        drop_nan=False,
        sample_size=20000,
        split='all',
        kfold=0,
    ):
        super().__init__(
            image_dir,
            labels_file,
            camera_file=None,
            transform=transform,
            feature_name=feature_name,
            drop_nan=drop_nan,
            sample_size=sample_size,
            split=split,
            kfold=kfold,
        )

    def preprocess_labels(self, df):
        return df

    def get_ids(self, split, kfold):
        """Get participant ids for the chosen split."""
        assert (
            (kfold == 0) and (split in ['train', 'val', 'test', 'all', 'all_holdout'])
        ) or ((kfold in list(range(1, 6))) and (split in ['train', 'val', 'test'])), (
            'Invalid combination of split and kfold.'
        )

        splits_file = os.path.join(os.path.dirname(self.labels_file), 'splits.json')
        with open(splits_file, 'r') as f:
            splits = json.load(f)

        if (split == 'all') or (split == 'all_holdout'):
            ids = np.array(
                splits['0']['train'] + splits['0']['val'] + splits['0']['test']
            )
        elif split == 'test':
            ids = np.array(splits['0']['test'])
        else:
            ids = np.array(splits[str(kfold)][split])

        ids = ids.astype(str)
        ids.sort()
        return ids


class APTOS(IDRiD):
    """
    Loads images from APTOS pre-processed dataset.
    """

    FEATURE_MAP = {
        'age': 'age',  # temporary
        'diagnosis': 'dr',
    }

    CAT_FEAT = ['dr']


class MESSIDOR(IDRiD):
    """
    Loads images from MESSIDOR pre-processed dataset.
    """

    FEATURE_MAP = {
        'age': 'age',  # temporary
        'adjudicated_dr_grade': 'dr',
        'adjudicated_dme': 'dme',
        'adjudicated_gradable': 'quality',
    }

    CAT_FEAT = ['dr', 'dme', 'quality']


class DeepDRiD(IDRiD):
    """
    Loads images from DeepDRiD pre-processed dataset.
    """

    FEATURE_MAP = {
        'age': 'age',  # temporary
        'dr': 'dr',
    }

    CAT_FEAT = ['dr']


class GLAUCOMA(IDRiD):
    """
    Loads images from Glaucom pre-processed dataset.
    """

    FEATURE_MAP = {
        'age': 'age',  # temporary
        'label': 'glaucoma',
    }

    CAT_FEAT = ['glaucoma']


class PAPILA(IDRiD):
    """
    Loads images from Glaucom pre-processed dataset.
    """

    FEATURE_MAP = {
        'Age': 'age',
        'Gender': 'gender',
        'Diagnosis': 'glaucoma',
        # 'glaucomabin': 'glaucomabin',
    }

    CAT_FEAT = ['glaucoma', 'glaucomabin', 'gender']


class FIVES(IDRiD):
    """
    Loads images from FIVES pre-processed dataset.
    """

    FEATURE_MAP = {'age': 'age', 'Disease': 'disease'}

    CAT_FEAT = ['disease']


class IDRiDSegmentation(IDRiD):
    """
    Loads images and masks from the IDRiD segmentation dataset.
    """

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str, Path): path to the image directory.
            transform (object): image transform.
        """
        self.image_dir = image_dir
        self.transform = transform

        #  Get image paths
        self.image_paths = os.listdir(os.path.join(self.image_dir, 'images'))
        self.image_paths = [
            os.path.join(self.image_dir, 'images', os.path.basename(p))
            for p in self.image_paths
        ]
        self.mask_paths = [
            os.path.join(self.image_dir, 'combined_masks', os.path.basename(p))
            for p in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # All images correspond to the positive class
        label = int(1)

        return image, mask, label


def load_dataset(
    dataset_dir='./datasets',
    dataset_name='imagenette',
    transform=None,
    image_size=(224, 224),
    feature_name='age',
    drop_nan=False,
    sample_size=20000,
    split='all',
    kfold=0,
):
    # Each class that corresponds to a dataset
    dataset_classes = {
        'eyepacs': EyePACS,
        'areds': AREDS,
        'ukb': UKB,
        'idrid': IDRiD,
        'deepdrid': DeepDRiD,
        'aptos': APTOS,
        'messidor': MESSIDOR,
        'glaucoma': GLAUCOMA,
        'papila': PAPILA,
        'fives': FIVES,
    }

    if dataset_name == 'cifar10':
        dataset_train, dataset_test, mapping = load_cifar10(transform)
        dataset = dataset_train if split == 'train' else dataset_test
    elif dataset_name == 'imagenette':
        dataset_train, dataset_test, mapping = load_imagenette(transform, image_size)
        dataset = dataset_train if split == 'train' else dataset_test
    elif dataset_name == 'eyepacs':
        dataset = EyePACS(
            image_dir=os.path.join(dataset_dir, dataset_name, f'dim_{image_size[0]}'),
            labels_file=os.path.join(
                dataset_dir, dataset_name, 'metadata', 'metadata.csv'
            ),
            camera_file=os.path.join(
                dataset_dir, dataset_name, 'metadata', 'all_camera.csv'
            ),
            transform=transform,
            feature_name=feature_name,
            drop_nan=drop_nan,
            sample_size=sample_size,
            split=split,
            kfold=kfold,
        )
        mapping = dataset.mappings
    elif dataset_name in dataset_classes.keys():
        dataset = dataset_classes[dataset_name](
            image_dir=os.path.join(dataset_dir, dataset_name, f'dim_{image_size[0]}'),
            labels_file=os.path.join(
                dataset_dir, dataset_name, 'metadata', 'metadata.csv'
            ),
            transform=transform,
            feature_name=feature_name,
            drop_nan=drop_nan,
            sample_size=sample_size,
            split=split,
            kfold=kfold,
        )
        mapping = dataset.mappings
    else:
        raise Exception(f'Dataset with name {dataset_name} is not supported.')

    return dataset, mapping


def load_dataset_all(
    dataset_dir='./dataset',
    dataset_name='imagenette',
    transform=None,
    image_size=(224, 224),
    feature_name='age',
    drop_nan=False,
    sample_size=20000,
):
    """Load a dataset or a collection of datasets without any splits. Useful for SSL pre-training."""

    if dataset_name in ['cifar10', 'imagenette']:
        dataset_train, mapping = load_dataset(
            dataset_dir, dataset_name, transform, image_size, split='train'
        )
        dataset_test, mapping = load_dataset(
            dataset_dir, dataset_name, transform, image_size, split='test'
        )
        dataset = ConcatDataset([dataset_train, dataset_test])

    elif dataset_name == 'all':
        # Load all eye datasets
        dataset_names = ['eyepacs', 'areds', 'ukb']
        sample_size = (
            sample_size if sample_size is None else sample_size // len(dataset_names)
        )
        datasets, mapping = [], {}
        for name in dataset_names:
            dataset_, mapping_ = load_dataset(
                dataset_dir=dataset_dir,
                dataset_name=name,
                transform=transform,
                image_size=image_size,
                feature_name=feature_name,
                drop_nan=drop_nan,
                sample_size=sample_size,
                split='all',
            )

            datasets.append(dataset_)
            mapping[name] = mapping_

        dataset = ConcatDataset(datasets)

    else:
        dataset, mapping = load_dataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            transform=transform,
            image_size=image_size,
            feature_name=feature_name,
            drop_nan=drop_nan,
            sample_size=sample_size,
            split='all',
        )

    return dataset, mapping


def get_stratify(df, col):
    """Get stratification from a data frame column, by patient id."""
    if col is not None:
        grouped = df.groupby('patient_id')[col].max()
        ids, stratify = grouped.index.to_numpy(), grouped.values
    else:
        ids = df['patient_id'].unique()
        stratify = None

    return ids, stratify


def generate_splits(metadata_file, testval_size=0.2, stratify_col=None, seed=42):
    """Create a json file with train-val-test and 5-fold splits for the participant ids."""
    df = pd.read_csv(metadata_file)

    fixed_test_datasets = ['idrid', 'fives']
    fixed_split_datasets = ['deepdrid']

    pretrain_datasets = ['eyepacs', 'areds', 'ukb']
    if any(pretrain_dataset in metadata_file for pretrain_dataset in pretrain_datasets):
        # Use holdout ids for pretrain datasets
        holdout_file = os.path.join(os.path.dirname(metadata_file), 'holdout.json')

        with open(holdout_file, 'r') as f:
            ids = np.array(json.load(f)['holdout'])

    elif any(name in metadata_file for name in fixed_test_datasets):
        # Fixed test set, get train/val from the train split
        do_testval_split = False
        ids_test = df[df['split'] == 'test']['patient_id'].unique()

        ids_kfold, stratify = get_stratify(df[df['split'] == 'train'], stratify_col)

        ids_train, ids_val = train_test_split(
            ids_kfold,
            test_size=testval_size / 2,
            shuffle=True,
            stratify=stratify,
            random_state=seed,
        )

    elif any(name in metadata_file for name in fixed_split_datasets):
        # Fixed train-val-test split
        do_testval_split = False
        ids_train = df[df['split'] == 'train']['patient_id'].unique()
        ids_val = df[df['split'] == 'val']['patient_id'].unique()
        ids_test = df[df['split'] == 'test']['patient_id'].unique()

        ids_kfold, stratify = get_stratify(
            df[df['split'].isin(['train', 'val'])], stratify_col
        )
    else:
        do_testval_split = True
        ids, stratify = get_stratify(df, stratify_col)

    splits = {}
    # Generate train, val and test splits (80-10-10)
    if do_testval_split:
        ids_train, ids_test = train_test_split(
            ids,
            test_size=testval_size,
            shuffle=True,
            stratify=stratify,
            random_state=seed,
        )
        if stratify is None:
            stratify2 = None
        else:
            stratify2 = stratify[get_matching_index(ids, ids_test)]

        ids_val, ids_test = train_test_split(
            ids_test, test_size=0.5, shuffle=True, stratify=stratify2, random_state=seed
        )

        ids_kfold = np.concatenate((ids_train, ids_val), axis=0)

        if stratify is not None:
            stratify = stratify[get_matching_index(ids, ids_kfold)]

    splits[0] = {
        'train': ids_train.tolist(),
        'val': ids_val.tolist(),
        'test': ids_test.tolist(),
    }

    print('train: ', ids_train.shape, 'val: ', ids_val.shape, 'test: ', ids_test.shape)

    # Generate 5 fold splits
    if stratify is None:
        kfold = KFold(5, shuffle=True, random_state=seed)
        kfold_splits = kfold.split(ids_kfold)
    else:
        kfold = StratifiedKFold(5, shuffle=True, random_state=seed)
        kfold_splits = kfold.split(ids_kfold, stratify)

    for i, (train_idx, val_idx) in enumerate(kfold_splits, start=1):
        splits[i] = {
            'train': ids_kfold[train_idx].tolist(),
            'val': ids_kfold[val_idx].tolist(),
        }

    print('train: ', train_idx.shape, 'val: ', val_idx.shape)

    splits_file = os.path.join(os.path.dirname(metadata_file), 'splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits, f)


def generate_holdout(metadata_file, destination_dir, testval_size=5000, seed=42):
    """Generate hold-out set for the pre-training datasets."""
    df = pd.read_csv(metadata_file)

    if 'eyepacs' in metadata_file:
        df = df[df['image_field'].isin(['field 1', 'field 2', 'field 3'])]
        df = df[df['session_image_quality'].isin(['Good', 'Excellent', 'Adequate'])]
    else:
        df = df[df['quality_label'] == 1]

    ids = df['patient_id'].unique()

    # Generate hold-out set for finetuning
    ids_pretrain, ids_holdout = train_test_split(
        ids, test_size=testval_size, shuffle=True, random_state=seed
    )

    holdout = {'pretrain': ids_pretrain.tolist(), 'holdout': ids_holdout.tolist()}
    holdout_file = os.path.join(destination_dir, 'holdout.json')
    with open(holdout_file, 'w') as f:
        json.dump(holdout, f)


def get_dataset_normalization(dataloader, device, n_channels=3):
    """Compute normalization parameters for a datasets. Assumes all images are tensors."""
    n_pixels, n_samples = 0, 0
    sum_per_channel = torch.zeros(n_channels, device=device)
    squared_sum_per_channel = torch.zeros(n_channels, device=device)
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        n_pixels = images.shape[0] * images.shape[2] * images.shape[3]
        sum_per_channel += images.sum(dim=(0, 2, 3))
        squared_sum_per_channel += images.pow(2).sum(dim=(0, 2, 3))
        n_samples += n_pixels

    dataset_mean = sum_per_channel / n_samples
    dataset_sd = torch.sqrt(
        (squared_sum_per_channel / n_samples) - (dataset_mean.pow(2))
    )

    return dataset_mean.tolist(), dataset_sd.tolist()


if __name__ == '__main__':
    pass
