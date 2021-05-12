from collections import Counter, defaultdict
import itertools
import logging
import os
import random

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from stratification.classification.datasets.base import GEORGEDataset


class ISICDataset(GEORGEDataset):
    """ISIC Dataset"""

    _channels = 3
    _resolution = 224
    _normalization_stats = {'mean': (0.71826, 0.56291, 0.52548), 'std': (0.16318, 0.14502, 0.17271)}
    superclass_names = ['benign', 'malignant']

    def __init__(
        self, root, split, transform=None, download=False, ontology='patch', augment=False
    ):
        assert transform is None
        transform = get_transform_ISIC(augment=augment)
        super().__init__(
            'isic', root, split, transform=transform, download=download, ontology=ontology
        )

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'isic')

    def _check_exists(self):
        """Checks whether or not the isic labels CSV has been initialized."""
        return os.path.isfile(os.path.join(self.processed_folder, 'labels.csv')) and os.path.isdir(
            os.path.join(self.processed_folder, 'images')
        )

    def _download(self):
        """Raises an error if the raw dataset has not yet been downloaded."""
        raise ValueError('Run `isic_download.py` in order to download the ISIC dataset.')

    def _load_samples(self):
        """Loads the ISIC dataset from the processed_folder"""
        if self.ontology == 'patch':
            self.true_subclass_names = ['benign/no_patch', 'benign/patch', 'malignant']
        elif self.ontology == 'histopathology':
            self.true_subclass_names = ['benign/no_hist', 'benign/hist', 'malignant']
        else:
            raise ValueError(f'Ontology {self.ontology} not supported.')

        self.images_path = os.path.join(self.processed_folder, 'images')
        labels_path = os.path.join(self.processed_folder, 'labels.csv')

        self.df = pd.read_csv(labels_path)
        # throw out unknowns
        self.df = self.df.loc[self.df.benign_malignant.isin({'benign', 'malignant'})]

        # split dataset
        split_df = self.df[self.df['fold'] == self.split]
        logging.info(f'Using {len(split_df)} of {len(self.df)} images in split "{self.split}"')
        self.df = split_df
        self.df = self.df.set_index('Image Index')

        superclass_labels, true_subclass_labels, alt_subclass_labels = [], [], []
        for idx in range(len(self.df)):
            sup_label, sub_label = self._get_labels_from_id(idx, ontology=self.ontology)
            superclass_labels.append(sup_label)
            true_subclass_labels.append(sub_label)
            _, alt_sub_label = self._get_labels_from_id(
                idx, ontology=({'patch', 'histopathology'} - {self.ontology}).pop()
            )
            alt_subclass_labels.append(alt_sub_label)

        X = self.df.index.values
        Y_dict = {
            'superclass': torch.tensor(superclass_labels),
            'true_subclass': torch.tensor(true_subclass_labels),
            'alt_subclass': torch.tensor(alt_subclass_labels),
        }
        return X, Y_dict

    def _get_labels_from_id(self, idx, ontology):
        """Gets superclass and subclass label for a given example id,
        based on the ontology."""
        suplabel = self.df['Diagnosis'].iloc[idx].astype(int)
        assert suplabel in (0, 1)

        if ontology == 'patch':
            sublabel_raw = self.df['patch'].iloc[idx].astype(int)
            if suplabel == 0:  # benign
                sublabel = sublabel_raw
            else:
                sublabel = 2
        elif ontology == 'histopathology':
            sublabel_raw = self.df['diagnosis_confirm_type'].iloc[idx]
            if suplabel == 0:
                sublabel = int(sublabel_raw == 'histopathology')
            else:
                sublabel = 2

        return suplabel, sublabel

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (x: torch.Tensor, y: dict) where X is a tensor representing an image
                and y is a dictionary of possible labels.
        """
        # load the original image
        image_path = os.path.join(self.processed_folder, 'images', self.X[idx])
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        x = image

        y_dict = {name: label[idx] for name, label in self.Y_dict.items()}
        return x, y_dict


def get_transform_ISIC(augment=False):
    test_transform_list = [
        transforms.Resize(ISICDataset._resolution),
        transforms.CenterCrop(ISICDataset._resolution),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ISICDataset._normalization_stats['mean'],
            std=ISICDataset._normalization_stats['std'],
        ),
    ]
    if not augment:
        return transforms.Compose(test_transform_list)

    train_transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ] + test_transform_list

    return transforms.Compose(train_transform_list)
